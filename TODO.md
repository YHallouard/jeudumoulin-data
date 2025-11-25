# 🚀 TODO - Optimisations Training Performance

> **Date Revue Complète**: 2025-11-25
> **Objectif**: Réduire le temps d'entraînement de 33h → 1-3h (AlphaZero) & optimiser DQN
> **Gains estimés**: 10-100x speed-up total
> **Statut**: ⚠️ ANALYSE COMPLÈTE - PRÊT POUR IMPLÉMENTATION

---

## 🔍 RÉSUMÉ EXÉCUTIF

### Problèmes Critiques Identifiés
1. **AlphaZero**: MCTS séquentiel → 90% du temps d'entraînement
2. **Batching GPU**: Inexistant → 95% du GPU inutilisé
3. **Training Loops**: Inférences une par une au lieu de batch
4. **Data Augmentation**: 4x overhead mémoire/temps non nécessaire
5. **Mixed Precision**: Non implémenté → 2x slowdown potentiel

### Quick Wins Prioritaires (< 1 semaine)
1. **Batch Training Loop** (P0): 4-8x gain immédiat
2. **Early Stopping MCTS** (P1): 2-5x gain
3. **Data Augmentation On-the-Fly** (P1): 1.5x gain + 75% mémoire
4. **Embedding Cache** (P1): 1.2x gain

### Impact Estimé Total
- **Court terme** (1 semaine): 33h → 4-6h (6-8x)
- **Moyen terme** (2-3 semaines): 33h → 30min-1h (30-60x)
- **Long terme** (1-2 mois): 33h → 10-20min (100-200x)

---

## 📊 DIAGNOSTIC ACTUEL

### Architecture
```
ITERATION D'ENTRAÎNEMENT (~20 min/iter × 100 = 33h)
├─ Self-Play (Rust+Python): 15-20 min (80-90% du temps) ⚠️ BOTTLENECK
│  ├─ 10 épisodes × 50 coups × 3000 simulations MCTS
│  └─ = 1,500,000 inférences séquentielles (CPU↔MPS transfers)
└─ Training (Python): 2-3 min (10-20% du temps) ✅ Déjà optimisé
```

### 🔴 BOTTLENECKS CRITIQUES (AlphaZero)

#### 1. **Training Loop Non-Batché** (⭐⭐⭐⭐⭐⭐) - **PRIORITÉ 0**
**Fichier**: `src_python/agent/alphazero/_trainer.py:275-311`
```python
# PROBLÈME: Boucle sur batch_size au lieu de batch inference
for state_emb, legal_moves, policy_target, value_target in zip(...):
    policy_pred, value_pred = self.agent.model.policy_value(state_emb, legal_moves)
    # Une inférence à la fois! 64x plus lent que batch
```
**Impact**:
- Training 30 epochs × 10 batches × 64 samples = **19,200 appels séquentiels**
- Avec batching: 30 × 10 = **300 appels batched**
- **Gain attendu: 4-8x sur phase training (3-5 min → 30-60 sec)**

**Solution Immédiate**:
```python
# Batch tous les états ensemble
states_tensor = torch.stack([torch.tensor(s) for s in state_embeddings])
# Adapter policy_value pour batch processing
policy_preds, value_preds = self.agent.model.policy_value_batch(states_tensor, legal_moves_batch)
```

---

#### 2. **MCTS Séquentiel** (⭐⭐⭐⭐⭐) - **PRIORITÉ 1**
**Fichier**: `src_rust/search/mcts.rs:222-305`
```rust
for _sim_idx in 0..num_simulations {  // 3000 simulations
    let (action_probs, value) = agent.predict(&current_state);  // Appel Python!
    // Chaque simulation = 1 appel Python + transfert données
}
```
**Impact**:
- 10 épisodes × 50 coups × 3000 sims = **1,500,000 appels Python/Rust**
- Overhead GIL + sérialisation: ~0.2ms/call
- Transferts CPU↔MPS: ~0.5ms/call
- **Total: ~10-15 min par itération (75-80% du temps)**

**Solutions**:
- **Court terme**: Early stopping quand politique converge (2-5x)
- **Moyen terme**: Lazy batching (accumuler N états → 1 batch inference) (10-20x)
- **Long terme**: Modèle en Rust pur (tch-rs) (50-100x)

---

#### 3. **Data Augmentation 4x Overhead** (⭐⭐⭐⭐)
**Fichier**: `src_rust/training/self_play.rs:119-141`
```rust
let random_rotation = rand::thread_rng().gen_range(0..4);
let rotated_state = state.rotate(random_rotation);
// Stocke TOUTES les rotations dans replay buffer
// 4x mémoire + 4x temps d'écriture
```
**Impact**:
- Replay buffer: 10,000 examples × 4 rotations = 40,000 entries inutiles
- Temps écriture: ~1-2 min par itération
- **Gain: 1.5x + économie 75% mémoire**

**Solution**: Rotation on-the-fly pendant sampling
```python
class AugmentedDataset(Dataset):
    def __getitem__(self, idx):
        base_idx = idx // 4
        rotation = idx % 4
        example = self.buffer[base_idx]
        return rotate(example, rotation)  # Calcul à la demande
```

---

#### 4. **Policy Head - Masques Python Non-Vectorisés** (⭐⭐⭐⭐)
**Fichier**: `src_python/agent/alphazero/_conditional_policy.py:175-215`
```python
def _create_to_mask_given_from(self, legal_actions, from_position, device):
    mask = torch.full((24,), float("-inf"), device=device)
    to_positions = set()
    for action in legal_actions:  # Python loop!
        action_from = action[0] if action[0] is not None else 24
        if action_from == from_position:
            to_positions.add(action[1])
    for pos in to_positions:  # Autre Python loop!
        mask[pos] = 0.0
    return mask
```
**Impact**:
- Appelé pour chaque `unique_from` dans batch
- 20-30% du temps forward pass
- **Gain: 1.3-1.5x avec vectorisation pure torch**

**Solution**: `GatedConditionalPolicyHead` utilise `_create_batch_to_mask` (déjà dans code!)
```python
# Ligne 520-540: Déjà implémenté mais pas utilisé par défaut!
def _create_batch_to_mask(self, legal_actions, device):
    mask = torch.full((num_actions, 24), float("-inf"), device=device)
    # Vectorized filling
```

---

### 🟡 BOTTLENECKS MOYENS (AlphaZero)

#### 5. **Embedding Lookup Répété** (⭐⭐⭐)
**Fichier**: `src_python/agent/alphazero/_conditional_policy.py:130`
```python
from_embs = self.position_embedding(from_indices_tensor)
# Lookup à chaque forward pour les mêmes 25 positions
```
**Impact**: 5-10% temps forward
**Solution**: Cache pré-calculé (1 ligne de code)
```python
self.register_buffer('pos_cache', self.position_embedding(torch.arange(25)))
```

---

#### 6. **Pas de Mixed Precision** (⭐⭐⭐)
**Fichiers**: `_trainer.py` (AlphaZero et DQN)
```python
# Aucune utilisation de autocast/GradScaler
policy_pred, value_pred = self.agent.model.policy_value(...)
loss.backward()  # Full FP32
```
**Impact**:
- 30-50% plus lent que FP16
- 2x plus de mémoire GPU
- **Gain: 1.3-1.5x + permet batch size 2x plus gros**

**Solution**: PyTorch AMP (5 lignes)
```python
from torch.amp import autocast, GradScaler
scaler = GradScaler('cuda')  # Ou 'mps' si supporté
with autocast('cuda'):
    policy, value = model(...)
```

---

### 🟢 BOTTLENECKS MINEURS

#### 7. **Scheduler Learning Rate Agressif** (⭐⭐)
**Fichier**: `config/train_alphazero.yaml:43-48`
```yaml
lr_scheduler_config:
  model_type: cosine_warm_rest_lr
  T_0: 10  # Restart tous les 10 epochs
  min_lr: 5e-5
```
**Impact**: Convergence parfois instable
**Solution**: Augmenter T_0 à 20-30 ou utiliser StepLR simple

---

### 🔴 BOTTLENECKS CRITIQUES (DQN)

#### 8. **Q-Value Computation Séquentiel** (⭐⭐⭐⭐⭐)
**Fichier**: `src_python/agent/dqn/_trainer.py:343-349`
```python
next_q_values = torch.zeros(batch_size, device=self.device)
for i in range(batch_size):  # Boucle sur batch!
    if dones[i] < 0.5:
        sample_actions = torch.randint(0, 25, (10, 3), device=self.device)
        sample_q = self.agent.model.predict_q_values(next_states[i], sample_actions)
        next_q_values[i] = sample_q.max()
```
**Impact**:
- Batch_size=128 → 128 appels séquentiels
- **Gain: 10-20x avec batch processing**

**Solution**:
```python
# Vectoriser complètement
sample_actions = torch.randint(0, 25, (batch_size, num_samples, 3))
q_values = self.agent.model.predict_q_values_batch(next_states, sample_actions)
next_q_values = q_values.max(dim=1)[0]
```

---

#### 9. **Opponent Random Non-Optimisé** (⭐⭐⭐)
**Fichier**: `src_python/agent/dqn/_trainer.py:307`
```python
if opponent == "random":
    move = random.choice(legal_moves)
# Python random.choice très lent vs Rust
```
**Impact**: 5-10% temps épisode
**Solution**: Déléguer à Rust

---

## 🎯 DÉCOUVERTES IMPORTANTES

### ✅ Points Forts Actuels
1. **Architecture Rust/Python**: Bien conçue, game logic rapide
2. **MCTS Réutilisation**: Déjà implémenté (ligne 183-190 mcts.rs)
3. **GatedConditionalPolicyHead**: Déjà vectorisé! (ligne 520-567)
4. **Replay Buffer**: Design solide, juste besoin on-the-fly augmentation
5. **MLflow Logging**: Excellent monitoring
6. **Modularité**: Facile d'ajouter optimisations progressivement

### ⚠️ Code Mort / Redondance
1. **3 Policy Heads**: Semi/Fully/Gated mais seul Semi utilisé par défaut
   - **Action**: Utiliser Gated par défaut (déjà vectorisé)
2. **DQN Prioritized Replay**: Implémenté mais jamais utilisé
   - **Action**: Tester avec `use_prioritized_replay: true`
3. **Tests Incomplets**: Beaucoup de test files mais pas exécutés
   - **Action**: CI/CD avec pytest

---

## 🎯 ROADMAP D'OPTIMISATION (RÉVISÉE)

### ✅ PHASE 0: Déjà Fait
- [x] Fix adjacency matrix device compatibility (MPS)
- [x] Diagnostic complet de performance
- [x] Identification des bottlenecks
- [x] Review complète du codebase (25/11/2025)

---

### 🔥 PHASE 1: QUICK WINS (2-3 jours) - Gain: 6-10x

#### ⬜ **P0: Batch Training Loop** (CRITIQUE - À FAIRE EN PREMIER)
**Impact**: ⭐⭐⭐⭐⭐⭐ | **Gain**: 4-8x | **Difficulté**: Moyenne | **Temps**: 4-6h

**Fichiers à Modifier**:
1. `src_python/agent/alphazero/_trainer.py:275-311`
2. `src_python/agent/alphazero/_models.py:48-53`
3. `src_python/agent/alphazero/_conditional_policy.py`

**Plan d'Implémentation**:

**Étape 1**: Ajouter méthode batch au modèle
```python
# Dans MLPDualNet (_models.py:48-53)
def policy_value_batch(
    self,
    state_embeddings_batch: torch.Tensor,  # [batch_size, 77]
    legal_moves_batch: list[list[list[int | None]]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batch inference pour training.

    Returns:
        policy_batch: [batch_size, max_num_actions] avec padding
        value_batch: [batch_size, 1]
    """
    state_features = self.backbone(state_embeddings_batch)  # [B, hidden]

    # Value: déjà batché naturellement
    values = self.value_head(state_features)  # [B, 1]

    # Policy: itérer mais sur GPU
    policies = []
    for i, legal_moves in enumerate(legal_moves_batch):
        policy = self.policy_head(state_features[i], legal_moves)
        policies.append(policy)

    return policies, values
```

**Étape 2**: Modifier training loop
```python
# Dans AlphaZeroTrainer._train_on_batch (ligne 275-311)
def _train_on_batch(self, state_embeddings, legal_moves_batch, policy_targets, value_targets):
    self.optimizer.zero_grad()

    # Convertir en tenseurs
    states_tensor = torch.stack([
        torch.tensor(s, device=self.device) for s in state_embeddings
    ])

    # Batch inference!
    policy_preds, value_preds = self.agent.model.policy_value_batch(
        states_tensor, legal_moves_batch
    )

    # Compute losses
    total_loss = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0

    for i in range(len(state_embeddings)):
        policy_target = torch.tensor(policy_targets[i], device=self.device)
        value_target = torch.tensor([value_targets[i]], device=self.device)

        p_loss = conditional_cross_entropy(policy_preds[i], policy_target)
        v_loss = F.mse_loss(value_preds[i], value_target)

        total_loss += p_loss + v_loss
        policy_loss_sum += p_loss.item()
        value_loss_sum += v_loss.item()

    total_loss.backward()
    self.optimizer.step()

    return policy_loss_sum / len(state_embeddings), ...
```

**Validation**:
- [ ] Temps training avant: ~3 min
- [ ] Temps training après: ~30-45 sec
- [ ] Pas de régression loss
- [ ] Utilisation GPU: 20% → 40-50%

---

#### ⬜ **P1.1: Utiliser GatedConditionalPolicyHead** (DÉJÀ IMPLÉMENTÉ!)
**Impact**: ⭐⭐⭐⭐ | **Gain**: 1.3-1.5x | **Difficulté**: Triviale | **Temps**: 5 min

**Problème**: Config utilise `SemiConditionalPolicyHead` avec loops Python
**Solution**: `GatedConditionalPolicyHead` déjà vectorisé (ligne 415-581)!

**Action**:
```yaml
# Dans config/train_alphazero.yaml:18
policy_head:
  model_type: gated_conditional_policy_head  # Changer de semi → gated
  state_embedding_dim: 128
  embedding:
    model_type: positional_embedding
    embedding_dim: 32
  hidden_dim: 512
  dropout_rate: 0.1
```

**Différence**:
- Semi: 3 loops Python pour créer masques (lignes 144-163, 299-320)
- Gated: Vectorisé avec `_create_batch_to_mask` (lignes 520-567)

**Validation**:
- [ ] Forward pass: 100ms → 60-70ms
- [ ] Loss identique
- [ ] Politique convergence similaire

---

#### ⬜ **P1.2: Early Stopping MCTS avec Convergence**
**Impact**: ⭐⭐⭐⭐⭐ | **Gain**: 2-5x | **Difficulté**: Moyenne | **Temps**: 3-4h

**Principe**: Arrêter MCTS quand la politique converge
- Implémenter détection de dominance: `max_visits / total_visits > 0.65`
- Config: min_sims=500, max_sims=3000, check_every=100
- Fichiers: `src_rust/search/mcts.rs`

**Approche Recommandée**: Dominance Threshold (simple)
```rust
pub fn is_dominated(&self, threshold: f64) -> bool {
    let max_visits = self.children.values().map(|c| c.visit_count).max().unwrap_or(0);
    let total_visits: usize = self.children.values().map(|c| c.visit_count).sum();
    (max_visits as f64 / total_visits as f64) > threshold
}
```

**Validation**:
- [ ] Mesurer distribution des convergences (avg sims utilisées)
- [ ] Vérifier pas de dégradation qualité (win rate)
- [ ] Profiler gain de temps réel

---

#### ⬜ **P1.3: Augmentation On-The-Fly**
**Impact**: ⭐⭐⭐⭐ | **Gain**: 1.3-1.5x + 75% mémoire | **Difficulté**: Faible | **Temps**: 2-3h

**Principe**: Générer rotations pendant le training, pas le self-play
- Fichiers: `src_rust/training/self_play.rs`, `src_python/agent/alphazero/_trainer.py`
- Divise par 4 la mémoire du replay buffer
- Divise par 4 le temps d'écriture des exemples

**Implémentation**:
```python
class AugmentedDataset:
    def __getitem__(self, idx):
        base_idx = idx // 4
        rotation = idx % 4
        example = self.base_examples[base_idx]
        return self._rotate_on_the_fly(example, rotation)
```

---

#### ⬜ **P1.4: Cache Position Embeddings**
**Impact**: ⭐⭐⭐ | **Gain**: 1.1-1.2x | **Difficulté**: Très faible | **Temps**: 15 min

**Fichier**: `src_python/agent/alphazero/_conditional_policy.py:130`

```python
# Dans SemiConditionalPolicyHead.__init__ (ligne 66)
def __init__(self, config):
    super().__init__()
    self.position_embedding = get_embedding(config.embedding)

    # AJOUTER: Pre-cache tous les embeddings
    self.register_buffer(
        'position_embeddings_cache',
        self.position_embedding(torch.arange(25))
    )

# Dans forward (ligne 130)
# AVANT: from_embs = self.position_embedding(from_indices_tensor)
# APRÈS:
from_embs = self.position_embeddings_cache[from_indices_tensor]
```

---

#### ⬜ **P1.5: DQN Batch Q-Value Computation**
**Impact**: ⭐⭐⭐⭐⭐ | **Gain**: 10-20x training DQN | **Difficulté**: Moyenne | **Temps**: 3h

**Fichier**: `src_python/agent/dqn/_trainer.py:343-349`

```python
# AVANT (séquentiel)
next_q_values = torch.zeros(self.batch_size, device=self.device)
for i in range(self.batch_size):
    if dones[i] < 0.5:
        sample_actions = torch.randint(0, 25, (10, 3))
        sample_q = self.agent.model.predict_q_values(next_states[i], sample_actions)
        next_q_values[i] = sample_q.max()

# APRÈS (batché)
num_samples = 10
sample_actions = torch.randint(0, 25, (self.batch_size, num_samples, 3), device=self.device)

# Expand next_states: [B, 77] → [B, num_samples, 77]
next_states_expanded = next_states.unsqueeze(1).expand(-1, num_samples, -1)
next_states_flat = next_states_expanded.reshape(-1, 77)
sample_actions_flat = sample_actions.reshape(-1, 3)

# Batch inference
q_values_flat = self.agent.model(next_states_flat, sample_actions_flat)
q_values = q_values_flat.reshape(self.batch_size, num_samples)

# Max over samples
next_q_values = q_values.max(dim=1)[0]
next_q_values = next_q_values * (1 - dones)  # Zero out terminal states
```

---

#### ⬜ **P1.6: Configuration Hyperparamètres**
**Impact**: ⭐⭐ | **Gain**: 1.3-1.5x | **Difficulté**: Triviale | **Temps**: 5 min

**Fichier**: `config/train_alphazero.yaml`

```yaml
training:
  epochs: 20  # 30 → 20 (avec batching, moins epochs suffisent)
  simulations: 2000  # 3000 → 2000 (en attendant early stopping)
  batch_size: 128  # 64 → 128 (batch plus gros si GPU le permet)

lr_scheduler_config:
  model_type: step_lr  # Plus stable que cosine_warm_rest
  step_size: 15
  gamma: 0.5
```

---

**📈 Gains Cumulés Phase 1**:
- **AlphaZero**: 6-10x total (20 min → 2-3 min/iter)
- **DQN**: 10-15x (training beaucoup plus rapide)

---

### 🚀 PHASE 2: OPTIMISATIONS MAJEURES (1-2 semaines) - Gain: 20-40x

#### ⬜ **P2.1: Parallélisation des Épisodes (Multi-Processing)**
**Impact**: ⭐⭐⭐⭐⭐ | **Gain**: 4-8x | **Difficulté**: Moyenne-Élevée | **Temps**: 1-2 jours

**Principe**: Exécuter plusieurs épisodes en parallèle (vrais processus, pas threads)
- Fichier: `src_python/agent/alphazero/_trainer.py`
- Utiliser `ProcessPoolExecutor` ou `joblib`
- Évite le GIL Python

**Approche Simple (joblib)**:
```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=4, backend='loky')(
    delayed(jdm_ru.execute_episode_rust)(
        self.agent, simulations, max_steps, temperature
    ) for _ in range(episodes_per_iteration)
)
```

**Contraintes**:
- [ ] Agent doit être picklable (sérialisation)
- [ ] Poids modèle partagés via fichier temporaire
- [ ] Pool réutilisable entre itérations

**Validation**:
- [ ] Mesurer speed-up réel (4 workers → 3-4x, 8 workers → 5-6x)
- [ ] Vérifier overhead mémoire acceptable
- [ ] Tester déterminisme (ordre épisodes)

---

#### ⬜ **P2.2: Batching MCTS - Lazy Evaluation**
**Impact**: ⭐⭐⭐⭐⭐⭐ | **Gain**: 10-30x | **Difficulté**: Élevée | **Temps**: 3-5 jours

**Principe**: Accumuler N états avant de faire batch inference
- Fichiers: `src_rust/search/mcts.rs`, `src_rust/agent/base.rs`
- Réduire appels Python de 3000 → 30 par coup

**Plan d'Implémentation**:

**Étape 1**: Ajouter batch_predict à Agent (Python)
```python
# src_python/agent/alphazero/_agent.py
def batch_predict(
    self,
    state_embeddings: list[list[float]],
    legal_moves_batch: list[list[list[int | None]]]
) -> tuple[list[dict[int, float]], list[float]]:
    """
    Batch inference pour MCTS.

    Returns:
        policies: Liste de dicts {action_idx: prob}
        values: Liste de floats
    """
    self.model.eval()

    # Convert to tensor
    states_tensor = torch.stack([
        torch.tensor(s, device=self.model.device) for s in state_embeddings
    ])

    # Batch inference
    policy_probs_list, values_tensor = self.model.policy_value_batch(
        states_tensor, legal_moves_batch
    )

    # Convert back to dicts
    policies = []
    for i, policy_probs in enumerate(policy_probs_list):
        policy_dict = {
            idx: float(prob)
            for idx, prob in enumerate(policy_probs.cpu().numpy())
        }
        policies.append(policy_dict)

    values = values_tensor.squeeze().cpu().numpy().tolist()
    return policies, values
```

**Étape 2**: Modifier MCTS Rust pour lazy evaluation
```rust
// src_rust/search/mcts.rs
pub struct MCTS {
    num_simulations: usize,
    batch_size: usize,  // NOUVEAU
}

impl MCTS {
    pub fn run(&self, agent: &dyn Agent, state: &Board) -> Node {
        let mut root = Node::new(0.0, 0);

        // Batch des simulations
        let num_batches = self.num_simulations / self.batch_size;

        for _batch_idx in 0..num_batches {
            let mut pending_states = Vec::with_capacity(self.batch_size);
            let mut pending_paths = Vec::with_capacity(self.batch_size);

            // Phase 1: Sélection (sans évaluation)
            for _ in 0..self.batch_size {
                let (leaf_state, path) = self.select_leaf_path(&mut root, state);
                pending_states.push(leaf_state);
                pending_paths.push(path);
            }

            // Phase 2: Batch inference (1 SEUL appel Python!)
            let state_embeddings: Vec<Vec<f32>> = pending_states
                .iter()
                .map(|s| s.to_embed())
                .collect();

            let legal_moves_batch: Vec<Vec<Vec<Option<usize>>>> = pending_states
                .iter()
                .map(|s| s.legal_moves().iter().map(|m| m.to_indices()).collect())
                .collect();

            let (policies, values) = agent.batch_predict(
                &state_embeddings,
                &legal_moves_batch
            );

            // Phase 3: Expansion + Backprop
            for i in 0..self.batch_size {
                self.expand_and_backprop(
                    &mut root,
                    &pending_paths[i],
                    &pending_states[i],
                    &policies[i],
                    values[i]
                );
            }
        }

        root
    }
}
```

**Étape 3**: Bindings PyO3
```rust
// src_rust/agent/base.rs
pub trait Agent: Send + Sync {
    fn predict(&self, state: &Board) -> (HashMap<Move, f32>, f32);

    // NOUVEAU
    fn batch_predict(
        &self,
        state_embeddings: &[Vec<f32>],
        legal_moves_batch: &[Vec<Vec<Option<usize>>>>
    ) -> (Vec<HashMap<usize, f32>>, Vec<f32>);
}

// Implémentation pour PythonAgent
impl Agent for PythonAgent {
    fn batch_predict(&self, state_embeddings, legal_moves_batch) -> ... {
        Python::with_gil(|py| {
            let result = self.agent
                .call_method1(py, "batch_predict", (state_embeddings, legal_moves_batch))
                .expect("Failed to call batch_predict");
            // Parse result
        })
    }
}
```

**Validation**:
- [ ] Tester convergence équivalente (KL divergence vs version séquentielle)
- [ ] Mesurer utilisation GPU (doit passer à 40-60%)
- [ ] Profiler gain réel (10-20x attendu)

---

#### ⬜ **P2.3: torch.compile sur Modèle**
**Impact**: ⭐⭐⭐⭐ | **Gain**: 1.5-2x | **Difficulté**: Moyenne | **Temps**: 2-3h

**Fichier**: `src_python/agent/alphazero/_models.py`

```python
# Dans MLPDualNet.__init__ (après ligne 40)
def __init__(self, config: MLPDualNetConfig):
    super().__init__()

    self.backbone = get_backbone(config.backbone)
    self.policy_head = get_policy_head(config.policy_head)
    self.value_head = nn.Sequential(...)

    # AJOUTER: Compile le modèle
    # Note: Compiler après premier forward (warmup)
    self._compiled = False

def _compile_if_needed(self):
    if not self._compiled and torch.cuda.is_available():
        # Compiler uniquement le backbone et value_head
        # Policy head a des dynamic shapes (legal_moves change)
        self.backbone = torch.compile(self.backbone, mode='reduce-overhead')
        self.value_head = torch.compile(self.value_head, mode='reduce-overhead')
        self._compiled = True

def policy_value(self, state_embedding, legal_moves):
    self._compile_if_needed()
    # Rest of code...
```

**Limitations Connues**:
- ⚠️ MPS (Apple Silicon) ne supporte pas torch.compile → utiliser CUDA
- ⚠️ Policy head difficile à compiler (dynamic shapes)
- ✅ Backbone et Value Head compilables facilement

**Alternative si pas CUDA**: Utiliser `torch.jit.script`
```python
self.backbone = torch.jit.script(self.backbone)
```

---

#### ⬜ **P2.4: Mixed Precision Training (FP16)**
**Impact**: ⭐⭐⭐⭐ | **Gain**: 1.3-1.8x + 50% mémoire | **Difficulté**: Faible | **Temps**: 1-2h

**Fichiers**:
- `src_python/agent/alphazero/_trainer.py:52-86`
- `src_python/agent/dqn/_trainer.py:56-79`

**AlphaZero**:
```python
# Dans AlphaZeroTrainer.__init__ (après ligne 86)
from torch.amp import autocast, GradScaler

self.use_amp = device in ['cuda', 'mps']  # MPS support expérimental
if self.use_amp:
    self.scaler = GradScaler(device)
else:
    self.scaler = None

# Dans _train_on_batch (ligne 275)
def _train_on_batch(self, ...):
    self.optimizer.zero_grad()

    total_loss = 0.0

    # Batch processing avec AMP
    if self.use_amp:
        with autocast(self.device):
            states_tensor = torch.stack([...])
            policy_preds, value_preds = self.agent.model.policy_value_batch(...)

            # Compute losses
            for i in range(len(state_embeddings)):
                p_loss = conditional_cross_entropy(policy_preds[i], ...)
                v_loss = F.mse_loss(value_preds[i], ...)
                total_loss += p_loss + v_loss

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
    else:
        # Fallback sans AMP
        # ... existing code ...
```

**DQN**: Similaire dans `_training_step`

**Notes**:
- ⚠️ MPS: Mixed precision support expérimental (PyTorch 2.4+)
- ✅ CUDA: Fully supported
- ✅ Batch size peut être doublé (même mémoire)

---

**📈 Gains Cumulés Phase 2**:
- **AlphaZero**: 30-60x total (20 min → 20-40 sec/iter)
- **DQN**: 40-80x (training très rapide)

---

### 🏆 PHASE 3: OPTIMISATIONS AVANCÉES (1-2 mois) - Gain: 50-200x

#### ⬜ **P3.1: Batching MCTS avec Virtual Loss**
**Impact**: ⭐⭐⭐⭐⭐ | **Gain**: 2x sur P2.2 | **Difficulté**: Très élevée | **Temps**: 1 semaine

**Principe**:
- Simulations vraiment parallèles (pas juste lazy batching)
- "Virtual loss": quand un thread sélectionne un nœud, ajoute loss temporaire
- Évite que tous les threads sélectionnent le même chemin
- Meilleure exploration + parallélisme thread-safe

**Référence**: MuZero paper (DeepMind)

**Implémentation**:
- Ajouter `virtual_loss: f64` à `Node`
- Lors sélection: `ucb_score + virtual_loss * thread_id`
- Backprop: retirer virtual loss, ajouter vraie value

**Difficulté**: Gestion concurrence, tuning hyperparamètres (alpha virtual loss)

---

#### ⬜ **P3.2: Modèle en Rust (tch-rs)**
**Impact**: ⭐⭐⭐⭐⭐⭐ | **Gain**: 30-100x | **Difficulté**: Très élevée | **Temps**: 2-3 semaines

**Principe**: Éliminer complètement Python de la boucle MCTS
- Charger checkpoint PyTorch en Rust avec `tch-rs`
- Inférence pure Rust (C++ LibTorch backend)
- Zero-copy, zero GIL, zero sérialisation

**Fichier**: `Cargo.toml` (déjà `tch = "0.18"` présent!)

**Plan**:
```rust
// src_rust/agent/torch_agent.rs (NOUVEAU)
use tch::{nn, Device, Tensor};

pub struct TorchAgent {
    model: nn::Sequential,
    device: Device,
}

impl TorchAgent {
    pub fn load(checkpoint_path: &str) -> Result<Self> {
        let device = Device::cuda_if_available();
        let vs = nn::VarStore::new(device);

        // Charger architecture
        let model = nn::seq()
            .add(nn::linear(&vs.root(), 77, 128, Default::default()))
            .add_fn(|x| x.relu())
            // ... rest of architecture
            ;

        // Charger poids depuis .safetensors
        vs.load(checkpoint_path)?;

        Ok(TorchAgent { model, device })
    }

    pub fn batch_predict(&self, states: &[Board]) -> (Vec<HashMap<Move, f32>>, Vec<f32>) {
        // Convert boards to tensor
        let state_embeddings: Vec<Vec<f32>> = states.iter()
            .map(|s| s.to_embed())
            .collect();

        let batch_tensor = Tensor::of_slice2(&state_embeddings)
            .to_device(self.device);

        // Forward pass (pure Rust/C++)
        let output = self.model.forward(&batch_tensor);

        // Parse output
        // ...
    }
}
```

**Avantages**:
- Latence: 0.5ms → 0.01ms par inference (50x)
- Throughput: 100 inferences/sec → 10,000/sec
- Zero overhead Python/Rust
- Peut tourner sur GPU sans Python

**Inconvénients**:
- Duplication de l'architecture (Python + Rust)
- Debugging plus difficile
- Maintenance 2x codebase

**Alternative**: `candle` (pure Rust, pas de C++ deps)

---

#### ⬜ **P3.3: Distributed Training (Ray)**
**Impact**: ⭐⭐⭐⭐⭐⭐ | **Gain**: Nx (N machines) | **Difficulté**: Très élevée | **Temps**: 3-4 semaines

**Principe**:
- Self-play distribué sur N workers
- Training centralisé sur GPU
- Shared replay buffer (Redis)
- Async updates

**Architecture**:
```
┌─────────────────────────────────────────┐
│           Ray Cluster                    │
├─────────────────────────────────────────┤
│ Worker 1: Self-Play (10 episodes/min)   │
│ Worker 2: Self-Play (10 episodes/min)   │
│ Worker 3: Self-Play (10 episodes/min)   │
│ ...                                      │
│ Worker N: Self-Play (10 episodes/min)   │
├─────────────────────────────────────────┤
│        Shared Redis Replay Buffer        │
├─────────────────────────────────────────┤
│ Trainer: Batch training on GPU           │
│  - Sample from Redis                     │
│  - Train & update weights                │
│  - Broadcast to workers                  │
└─────────────────────────────────────────┘
```

**Outils**:
- Ray (orchestration)
- Redis (replay buffer partagé)
- torch.distributed (multi-GPU training si besoin)

**Gain**: Linéaire avec nombre de workers (10 workers → 10x self-play speed)

---

#### ⬜ **P3.4: Curriculum Learning**
**Impact**: ⭐⭐⭐ | **Gain**: 2-3x sample efficiency | **Difficulté**: Moyenne | **Temps**: 1 semaine

**Principe**:
- Commencer avec adversaires faibles (random)
- Progressivement augmenter difficulté (past checkpoints)
- Final: self-play pur

**Config**:
```yaml
training:
  curriculum:
    enabled: true
    stages:
      - opponent: random
        iterations: 10
        simulations: 1000
      - opponent: checkpoint_10
        iterations: 20
        simulations: 1500
      - opponent: self
        iterations: 70
        simulations: 2000
```

---

**📈 Gains Cumulés Phase 3**:
- **AlphaZero**: 100-200x total (20 min → 5-12 sec/iter)
- **Training 100 iter**: 33h → 8-20 minutes

---

## 📈 ESTIMATIONS TEMPS D'ENTRAÎNEMENT

### AlphaZero (100 iterations)

| Phase | Temps/Iter | Total (100 iter) | Speed-up | Optimisations Clés |
|-------|------------|------------------|----------|--------------------|
| **Actuel** | 20 min | **33h 20min** | 1x | - |
| **Après Phase 1** | 2-3 min | **3-5h** ✅ | 6-10x | Batch Training + Gated Policy + Early Stop MCTS |
| **Après Phase 2** | 20-40 sec | **33-66 min** ✅✅ | 30-60x | + MCTS Batching + Mixed Precision |
| **Après Phase 3** | 5-12 sec | **8-20 min** ✅✅✅ | 100-200x | + Rust Model + Distributed |

### DQN (1000 episodes)

| Phase | Temps Total | Speed-up | Optimisations Clés |
|-------|-------------|----------|--------------------|
| **Actuel** | ~8-10h | 1x | - |
| **Après Phase 1** | ~45-60 min | 10-15x | Batch Q-Value Computation |
| **Après Phase 2** | ~20-30 min | 20-30x | + Mixed Precision + torch.compile |

---

## 🎯 PLAN D'ACTION IMMÉDIAT

### 🔴 PRIORITÉ CRITIQUE (Jour 1-2) - Gain: 6-10x
**Objectif**: Passer de 33h → 3-5h (AlphaZero)

#### Jour 1 Matin (4h)
- [x] **Review complète** (FAIT)
- [ ] **P0: Batch Training Loop** (4-6h)
  - Implémenter `policy_value_batch` dans MLPDualNet
  - Modifier `_train_on_batch` pour utiliser batching
  - Tester sur config light (5 iterations)
  - **Gain attendu**: 4-8x sur phase training

#### Jour 1 Après-midi (4h)
- [ ] **P1.1: Utiliser GatedConditionalPolicyHead** (5 min)
  - Changer config: `model_type: gated_conditional_policy_head`
  - **Gain attendu**: 1.3-1.5x

- [ ] **P1.4: Cache Position Embeddings** (15 min)
  - Ajouter buffer dans `__init__`
  - Remplacer lookup dans `forward`
  - **Gain attendu**: 1.1-1.2x

- [ ] **P1.2: Early Stopping MCTS** (3-4h)
  - Implémenter détection convergence dans Node
  - Ajouter config `min_simulations`, `convergence_threshold`
  - **Gain attendu**: 2-5x

#### Jour 2 Matin (3h)
- [ ] **P1.3: Data Augmentation On-the-Fly** (2-3h)
  - Modifier `self_play.rs` pour ne pas stocker rotations
  - Créer `AugmentedDataset` wrapper en Python
  - **Gain attendu**: 1.3-1.5x + 75% mémoire

- [ ] **P1.6: Config Hyperparamètres** (5 min)
  - epochs: 30 → 20
  - simulations: 3000 → 2000 (temporaire)
  - batch_size: 64 → 128
  - **Gain attendu**: 1.3x

#### Jour 2 Après-midi (2h)
- [ ] **Testing & Validation**
  - Run config light (5 iter): vérifier gains
  - Run config standard (10 iter): valider qualité
  - Mesurer win rate vs random
  - Documenter gains réels

**Résultat attendu Jour 2**:
- Temps/iter: 20 min → 2-3 min (6-10x ✅)
- Training 100 iter: 33h → 3-5h

---

### 🟡 PRIORITÉ HAUTE (Semaine 1) - Gain: 30-60x

#### Jour 3-4 (2 jours)
- [ ] **P2.1: Parallélisation Épisodes** (1-2 jours)
  - Implémenter avec `joblib` ou `ProcessPoolExecutor`
  - 4-8 workers selon CPU
  - **Gain attendu**: 3-5x sur self-play

#### Jour 5-7 (3 jours)
- [ ] **P2.2: MCTS Batching** (3-5 jours)
  - Ajouter `batch_predict` à Agent Python
  - Modifier MCTS Rust pour lazy evaluation
  - Bindings PyO3
  - **Gain attendu**: 10-20x sur MCTS

#### Jour 8 (1 jour)
- [ ] **P2.4: Mixed Precision** (1-2h)
  - AlphaZero trainer
  - DQN trainer
  - **Gain attendu**: 1.3-1.8x

- [ ] **P2.3: torch.compile** (2-3h)
  - Compiler backbone + value_head
  - **Gain attendu**: 1.5-2x

**Résultat attendu Semaine 1**:
- Temps/iter: 20 min → 20-40 sec (30-60x ✅✅)
- Training 100 iter: 33h → 33-66 min

---

### 🟢 PRIORITÉ MOYENNE (Semaines 2-3) - DQN

- [ ] **P1.5: DQN Batch Q-Value** (3h)
  - Vectoriser computation Q-values
  - **Gain attendu**: 10-20x DQN training

- [ ] **DQN Mixed Precision** (1h)

- [ ] **DQN Prioritized Replay** (déjà implémenté!)
  - Activer dans config: `use_prioritized_replay: true`

---

### ⚪ PRIORITÉ BASSE (Mois 2+) - Phase 3

- [ ] P3.1: Virtual Loss MCTS (1 semaine)
- [ ] P3.2: Rust Model (tch-rs) (2-3 semaines)
- [ ] P3.3: Distributed Training (3-4 semaines)
- [ ] P3.4: Curriculum Learning (1 semaine)

---

## 📋 CHECKLIST DE VALIDATION

### Pour Chaque Optimisation

#### Tests Fonctionnels
- [ ] Code compile sans erreur
- [ ] Tests unitaires passent
- [ ] Config light (5 iter) fonctionne
- [ ] Config standard (10 iter) fonctionne

#### Performance
- [ ] Temps avant/après mesuré (avec `time` ou MLflow)
- [ ] Speed-up réel vs théorique documenté
- [ ] Profiling CPU/GPU (avec `torch.profiler` si besoin)
- [ ] Utilisation mémoire stable

#### Qualité
- [ ] Loss converge normalement
- [ ] Win rate vs random ≥ baseline
- [ ] Pas de NaN dans losses
- [ ] Politique semble raisonnable (pas trop aléatoire)

#### Documentation
- [ ] Gains réels ajoutés dans ce fichier
- [ ] Code commenté si complexe
- [ ] Commit git avec message clair

---

## 🔧 CONFIGURATIONS DE TEST

### Config Ultra-Light (Dev rapide - 5 min)
```yaml
training:
  iterations: 3
  episodes: 3
  simulations: 500
  max_episode_steps: 50
  epochs: 5
  batch_size: 32
```
**Usage**: Tester que code fonctionne sans crash

### Config Light (Tests Perf - 30 min)
```yaml
training:
  iterations: 5
  episodes: 5
  simulations: 1000
  max_episode_steps: 150
  epochs: 10
  batch_size: 64
```
**Usage**: Mesurer gains de performance

### Config Standard (Validation - 2-3h)
```yaml
training:
  iterations: 10
  episodes: 10
  simulations: 2000
  max_episode_steps: 300
  epochs: 20
  batch_size: 128
```
**Usage**: Vérifier qualité du modèle

### Config Full (Production - 3-5h après optims)
```yaml
training:
  iterations: 100
  episodes: 10
  simulations: 2000  # avec early stopping
  max_episode_steps: 300
  epochs: 20  # réduit de 30
  batch_size: 128
```
**Usage**: Training final

---

## 📊 TRACKING DES GAINS RÉELS

### Baseline (2025-11-25)
**Config**: Standard (10 iter)
- **Temps total**: ? (à mesurer)
- **Temps/iter**: ~20 min (estimé basé sur config)
- **Device**: CPU self-play, MPS training
- **GPU Utilization**: ~5-10% (très faible)
- **Win rate vs Random**: TBD
- **Bottleneck**: MCTS séquentiel (75%) + Training non-batché (20%)

### Phase 1 - Résultats (Target: Jour 2)
- [ ] **Date**: ___________
- [ ] **Optimisations implémentées**: P0, P1.1, P1.2, P1.3, P1.4, P1.6
- [ ] **Temps/iter**: _____ min (target: 2-3 min)
- [ ] **Speed-up**: _____x (target: 6-10x)
- [ ] **GPU Utilization**: _____% (target: 30-40%)
- [ ] **Win rate vs Random**: _____% (doit être ≥ baseline)
- [ ] **Replay buffer size**: _____ MB (target: 25% du baseline)
- [ ] **Notes**: ___________

### Phase 2 - Résultats (Target: Fin Semaine 1)
- [ ] **Date**: ___________
- [ ] **Optimisations implémentées**: + P2.1, P2.2, P2.3, P2.4
- [ ] **Temps/iter**: _____ sec (target: 20-40 sec)
- [ ] **Speed-up**: _____x (target: 30-60x)
- [ ] **GPU Utilization**: _____% (target: 50-70%)
- [ ] **Win rate vs Random**: _____%
- [ ] **Notes**: ___________

### Phase 3 - Résultats (Target: Fin Mois 2)
- [ ] **Date**: ___________
- [ ] **Optimisations implémentées**: + P3.1, P3.2
- [ ] **Temps/iter**: _____ sec (target: 5-12 sec)
- [ ] **Speed-up**: _____x (target: 100-200x)
- [ ] **Notes**: ___________

---

## 🧪 COMMANDES DE BENCHMARK

### Mesurer Temps Training
```bash
# Baseline
time python -m jeudumoulin_py.cli.train \
  --config config/train_alphazero.yaml \
  --override training.iterations=5

# Après optimisations
time python -m jeudumoulin_py.cli.train \
  --config config/train_alphazero_optimized.yaml \
  --override training.iterations=5
```

### Profiling GPU
```python
# Ajouter dans trainer
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    self._train_on_batch(...)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Monitoring MLflow
```bash
# Lancer UI
mlflow ui --port 5001

# Comparer runs
# http://localhost:5001 → Compare metrics
```

---

## 💡 IDÉES FUTURES / BACKLOG

### Optimisations Algorithme (Post Phase 3)
- [ ] **Adaptive MCTS**: Plus de simulations dans positions critiques
  - Détecter positions "difficiles" (valeur incertaine)
  - Allouer dynamiquement budget simulations

- [ ] **Temperature Decay**: Commencer exploratoire, finir greedy
  - Temperature: 1.0 → 0.1 progressivement
  - Améliore stabilité fin training

- [ ] **Transposition Table**: Cache évaluations positions identiques
  - Hash board state
  - Réutiliser policy/value si déjà évalué
  - Gain potentiel: 2-3x si beaucoup de transpositions

### Optimisations Architecture
- [ ] **Two-Model Approach**:
  - Small fast model pour self-play (latence)
  - Large slow model pour training (qualité)
  - Knowledge distillation

- [ ] **Attention Mechanism**:
  - Remplacer GNN par Transformer
  - Peut mieux capturer relations board

- [ ] **Neural Architecture Search**:
  - Trouver meilleure config (hidden dims, layers)
  - Utiliser Optuna

### Infrastructure
- [ ] **Compression Replay Buffer**:
  - Quantize float32 → int8
  - Économie 4x mémoire

- [ ] **Checkpoint Streaming**:
  - Sauvegarder sur S3/GCS au lieu de disque local
  - Training interruptible

- [ ] **Multi-GPU Training**:
  - torch.distributed
  - Batch size encore plus gros

### DQN Specific
- [ ] **Dueling DQN**: Separate value/advantage streams
- [ ] **Double DQN**: Reduce overestimation bias
- [ ] **N-step Returns**: Use multi-step TD targets
- [ ] **Noisy Networks**: Exploration via network noise

### Expériences
- [ ] **Comparer backbones**: MLP vs GraphConv vs Transformer
- [ ] **Ablation study**: Impact de chaque optimisation
- [ ] **Scaling laws**: Performance vs compute (iter × sims × epochs)

---

## 📚 RÉFÉRENCES

### Papers
- AlphaZero: https://arxiv.org/abs/1712.01815
- MuZero: https://arxiv.org/abs/1911.08265 (batching MCTS)
- EfficientZero: https://arxiv.org/abs/2111.00210 (sample efficiency)

### Code
- Python multiprocessing best practices: https://docs.python.org/3/library/multiprocessing.html
- PyTorch compile: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- tch-rs: https://github.com/LaurentMazare/tch-rs

---

## 🐛 PROBLÈMES CONNUS & SOLUTIONS

### MPS Device (Apple Silicon)
**Status**: ⚠️ Limitations connues

- [x] ~~Adjacency matrix reste sur CPU~~ → **Fixé**
- [ ] **Mixed precision AMP**: Support expérimental (PyTorch 2.4+)
  - ✅ Peut fonctionner mais potentiellement instable
  - 🔧 Fallback: Utiliser CPU ou CUDA
  - 📝 Tester avec `try/except` dans code

- [ ] **torch.compile**: Pas supporté sur MPS (PyTorch 2.5)
  - ❌ `torch.compile()` lève erreur
  - 🔧 Solution: Check device avant compile
  ```python
  if device == 'cuda':
      self.model = torch.compile(self.model)
  ```
  - 📝 Alternative: `torch.jit.script` (partiel support)

### Sérialisation & Multiprocessing
**Status**: ⚠️ Workaround nécessaire

- [ ] **PyTorch models non-picklables**
  - Problème: `nn.Module` avec closures/lambdas
  - Impact: `ProcessPoolExecutor` peut crasher
  - 🔧 **Solution 1**: Passer poids via fichier temporaire
    ```python
    # Worker process
    def run_episode(checkpoint_path):
        agent = AlphaZeroAgent.load(checkpoint_path)
        return episode_data

    # Main process
    with tempfile.NamedTemporaryFile() as f:
        torch.save(agent.state_dict(), f.name)
        results = pool.map(run_episode, [f.name] * num_workers)
    ```
  - 🔧 **Solution 2**: Utiliser `torch.multiprocessing` (share tensors)
  - 🔧 **Solution 3**: Ray (gère sérialisation automatiquement)

### GIL Python
**Status**: ✅ Résolu avec multiprocessing

- [x] **Threads ne parallélisent pas** (GIL)
  - ❌ `ThreadPoolExecutor`: Pas de vrai parallélisme
  - ✅ `ProcessPoolExecutor`: Bypass GIL
  - ✅ `joblib` avec `backend='loky'`: Bypass GIL
  - 📝 Toujours utiliser processes, jamais threads pour CPU-bound

### MCTS Convergence
**Status**: ⚠️ À surveiller avec early stopping

- [ ] **Early stopping trop agressif**
  - Risque: Stop avant vraie convergence
  - 🔧 Solution: Tuner threshold (0.6-0.7) + min_simulations (500-1000)
  - 📝 Monitorer distribution sims utilisées

- [ ] **Virtual loss trop élevé**
  - Risque: Simulations évitent nœuds prometteurs
  - 🔧 Solution: Alpha virtual loss < 0.5
  - 📝 A/B test avec/sans virtual loss

### Memory Leaks
**Status**: ✅ Pas de leak connu, mais à surveiller

- [ ] **Replay buffer croissance**
  - ✅ `deque(maxlen=...)` gère automatiquement
  - 📝 Monitorer `torch.cuda.memory_allocated()` si GPU

- [ ] **MCTS tree accumulation**
  - ✅ Root node réinitialisé chaque coup
  - ✅ Garbage collector Python nettoie
  - 📝 Si leak: `gc.collect()` après épisodes

### Numerical Stability
**Status**: ✅ Robuste mais à vérifier

- [ ] **Log probabilities underflow**
  - ✅ Epsilon = 1e-8 dans `conditional_cross_entropy`
  - ✅ `torch.clamp` prevent 0 or 1
  - 📝 Surveiller NaN dans losses (MLflow)

- [ ] **MCTS value explosion**
  - ✅ Tanh value head: output ∈ [-1, 1]
  - ✅ UCB scores normalisés
  - 📝 Si problème: clip gradients (déjà fait)

---

## 📚 RÉFÉRENCES TECHNIQUES

### Papers
- **AlphaZero**: [Mastering Chess and Shogi by Self-Play with a General RL Algorithm](https://arxiv.org/abs/1712.01815)
- **MuZero**: [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265) - Batching MCTS
- **EfficientZero**: [Mastering Atari Games with Limited Data](https://arxiv.org/abs/2111.00210) - Sample efficiency
- **Prioritized Experience Replay**: [Schaul et al. ICLR 2016](https://arxiv.org/abs/1511.05952)

### Code & Libs
- **PyTorch Performance**: [Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- **torch.compile**: [Introduction Tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- **Mixed Precision**: [AMP Guide](https://pytorch.org/docs/stable/amp.html)
- **tch-rs**: [Rust LibTorch Bindings](https://github.com/LaurentMazare/tch-rs)
- **Ray**: [Distributed Framework](https://docs.ray.io/en/latest/)
- **joblib**: [Easy Parallel Python](https://joblib.readthedocs.io/)

### Blogs & Tutorials
- **AlphaZero from Scratch**: [David Foster Blog](https://applied-data.science/blog/alphazero-from-scratch/)
- **MCTS Batching**: [Parallel MCTS](https://arxiv.org/abs/1705.08926)
- **PyTorch Optimization**: [Tips and Tricks](https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/)

---

## 🏁 PROCHAINES ÉTAPES

### Immédiat (Aujourd'hui)
1. ✅ **Review complète** - FAIT
2. [ ] **Créer branch `feature/batch-training`**
3. [ ] **Implémenter P0: Batch Training Loop**
4. [ ] **Tester sur config ultra-light**

### Demain
1. [ ] **P1.1-P1.4**: Quick wins
2. [ ] **Mesurer baseline vs optimisé**
3. [ ] **Commit & PR**

### Cette Semaine
1. [ ] **Phase 1 complète**
2. [ ] **Documentation gains réels**
3. [ ] **Commencer Phase 2**

---

**Dernière mise à jour**: 2025-11-25 (Review Complète)
**Auteur**: AI Code Reviewer
**Status**: 📋 PRÊT POUR IMPLÉMENTATION - Phase 1 prioritaire
**Effort Phase 1**: 2 jours
**Impact Phase 1**: 6-10x speed-up (33h → 3-5h)
