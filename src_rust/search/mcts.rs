use std::collections::HashMap;

use indicatif::{ProgressBar, ProgressStyle};
use rand::distributions::WeightedIndex;
use rand::prelude::*;

use crate::agent::base::Agent;
use crate::game::{Board, Move};

const C_PUCT: f64 = 1.25;
const VIRTUAL_LOSS_COUNT: usize = 1;
const VIRTUAL_LOSS_VALUE: f64 = -1.0;

fn ucb_score(parent: &Node, child: &Node) -> f64 {
    let prior_score =
        child.prior * C_PUCT * (parent.visit_count as f64).sqrt() / (child.visit_count + 1) as f64;
    let value_score = if child.visit_count > 0 {
        -child.value()
    } else {
        0.0
    };
    value_score + prior_score
}

#[derive(Debug, Clone)]
pub struct Node {
    pub visit_count: usize,
    pub prior: f64,
    pub value_sum: f64,
    pub children: HashMap<Move, Node>,
    pub state: Option<Board>,
    pub depth: usize,
}

impl Node {
    pub fn new(prior: f64, depth: usize) -> Self {
        Node {
            visit_count: 0,
            prior,
            value_sum: 0.0,
            children: HashMap::new(),
            state: None,
            depth,
        }
    }

    pub fn expanded(&self) -> bool {
        !self.children.is_empty()
    }

    pub fn value(&self) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f64
        }
    }

    pub fn select_action(&self, temperature: f64) -> Move {
        let visit_counts: Vec<usize> = self.children.values().map(|c| c.visit_count).collect();
        let actions: Vec<Move> = self.children.keys().cloned().collect();

        if actions.is_empty() {
            panic!("No actions available to select");
        }

        if temperature == 0.0 {
            let max_idx = visit_counts
                .iter()
                .enumerate()
                .max_by_key(|(_, &count)| count)
                .map(|(idx, _)| idx)
                .unwrap();
            actions[max_idx].clone()
        } else if temperature.is_infinite() {
            let mut rng = thread_rng();
            actions.choose(&mut rng).unwrap().clone()
        } else {
            let visit_count_distribution: Vec<f64> = visit_counts
                .iter()
                .map(|&count| (count as f64).powf(1.0 / temperature))
                .collect();
            let sum: f64 = visit_count_distribution.iter().sum();

            if sum <= 0.0 || !sum.is_finite() {
                let mut rng = thread_rng();
                return actions.choose(&mut rng).unwrap().clone();
            }

            let normalized: Vec<f64> = visit_count_distribution.iter().map(|v| v / sum).collect();

            if normalized.iter().any(|&v| !v.is_finite() || v < 0.0) {
                let mut rng = thread_rng();
                return actions.choose(&mut rng).unwrap().clone();
            }

            let mut rng = thread_rng();
            match WeightedIndex::new(&normalized) {
                Ok(dist) => {
                    let idx = dist.sample(&mut rng);
                    actions[idx].clone()
                }
                Err(_) => actions.choose(&mut rng).unwrap().clone(),
            }
        }
    }

    pub fn select_child(&mut self) -> (Move, &mut Node) {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_action: Option<Move> = None;

        for (action, child) in &self.children {
            let score = ucb_score(self, child);
            if score > best_score {
                best_score = score;
                best_action = Some(action.clone());
            }
        }

        let action = best_action.expect("No best action found");
        let child = self.children.get_mut(&action).unwrap();
        (action, child)
    }

    pub fn expand(&mut self, state: Board, action_probs: HashMap<Move, f32>, inject_noise: bool) {
        self.state = Some(state);

        let mut final_probs = action_probs;

        if inject_noise {
            let alpha = 0.03;
            let epsilon = 0.25;
            let num_actions = final_probs.len();

            let mut rng = thread_rng();
            let gamma = rand_distr::Gamma::new(alpha, 1.0).unwrap();
            let noise: Vec<f64> = (0..num_actions).map(|_| gamma.sample(&mut rng)).collect();
            let sum: f64 = noise.iter().sum();
            let normalized_noise: Vec<f64> = noise.iter().map(|v| v / sum).collect();

            final_probs = final_probs
                .into_iter()
                .enumerate()
                .map(|(i, (action, prob))| {
                    let new_prob = (1.0 - epsilon) * prob + epsilon * normalized_noise[i] as f32;
                    (action, new_prob)
                })
                .collect();
        }

        for (action, prob) in final_probs {
            if prob != 0.0 {
                self.children
                    .insert(action, Node::new(prob as f64, self.depth + 1));
            }
        }
    }

    pub fn update_depth(&mut self, depth: usize) {
        self.depth = depth;
        for child in self.children.values_mut() {
            child.update_depth(depth + 1);
        }
    }
}

pub struct MCTS {
    num_simulations: usize,
    min_simulations: usize,
    temp_threshold: usize,
    show_progress: bool,
}

impl MCTS {
    pub fn new(num_simulations: usize, show_progress: bool) -> Self {
        MCTS {
            num_simulations,
            min_simulations: 100,
            temp_threshold: 50,
            show_progress,
        }
    }

    pub fn run(
        &self,
        agent: &dyn Agent,
        state: &Board,
        depth: usize,
        root: Option<Node>,
    ) -> Node {
        let mut root = if let Some(mut r) = root {
            let target_total = self.num_simulations;
            let already_done = r.visit_count;
            let nb_simulations = if already_done >= target_total {
                self.min_simulations
            } else {
                std::cmp::max(self.min_simulations, target_total - already_done)
            };

            r.prior = 0.0;
            r.update_depth(0);
            if !r.expanded() {
                let results = agent.predict(&[state]);
                let (action_probs, _) = results.into_iter().next().unwrap();
                r.expand(state.clone(), action_probs, true);
            }
            (r, nb_simulations)
        } else {
            let nb_simulations = self.num_simulations;
            let mut r = Node::new(0.0, depth);
            let results = agent.predict(&[state]);
            let (action_probs, _) = results.into_iter().next().unwrap();
            r.expand(state.clone(), action_probs, true);
            (r, nb_simulations)
        };

        let pb = if self.show_progress {
            let pb = ProgressBar::new(root.1 as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{msg} [{bar:40.cyan/blue}] {pos}/{len}")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            pb.set_message("MCTS Simulations");
            Some(pb)
        } else {
            None
        };

        for _sim_idx in 0..root.1 {
            let mut current_state = root.0.state.clone().unwrap();
            let mut search_path: Vec<Move> = vec![];
            let mut node_ref = &mut root.0 as *mut Node;

            unsafe {
                while !current_state.is_terminal() && (*node_ref).depth < self.temp_threshold {
                    if !(*node_ref).expanded() {
                        break;
                    }

                    let (action, child) = (*node_ref).select_child();
                    search_path.push(action.clone());
                    current_state = current_state.apply_move(&action);
                    node_ref = child as *mut Node;
                }

                let (value, action_probs) = if current_state.is_terminal() {
                    let winner = current_state.winner();
                    let value = match winner {
                        None => 0.0,
                        Some(w) if w == current_state.current_player => 1.0,
                        _ => -1.0,
                    };
                    (value, HashMap::new())
                } else if (*node_ref).depth < self.temp_threshold {
                    let results = agent.predict(&[&current_state]);
                    let (action_probs, value) = results.into_iter().next().unwrap();
                    (value as f64, action_probs)
                } else {
                    (0.0, HashMap::new())
                };

                if !current_state.is_terminal() && (*node_ref).depth < self.temp_threshold {
                    (*node_ref).expand(current_state.clone(), action_probs, false);
                }

                let current_node = node_ref;
                let mut current_value = value;

                (*current_node).value_sum += current_value;
                (*current_node).visit_count += 1;
                current_value = -current_value;

                for action in search_path.iter().rev() {
                    let mut parent_ref = &mut root.0 as *mut Node;
                    for prev_action in search_path.iter() {
                        if prev_action == action {
                            break;
                        }
                        if let Some(child) = (*parent_ref).children.get_mut(prev_action) {
                            parent_ref = child as *mut Node;
                        }
                    }

                    (*parent_ref).value_sum += current_value;
                    (*parent_ref).visit_count += 1;
                    current_value = -current_value;
                }

                root.0.value_sum += current_value;
                root.0.visit_count += 1;
            }

            if let Some(ref pb) = pb {
                pb.inc(1);
            }
        }

        if let Some(pb) = pb {
            pb.finish_with_message("MCTS completed");
        }

        root.0
    }

    pub fn run_batched(
        &self,
        agent: &dyn Agent,
        state: &Board,
        depth: usize,
        root: Option<Node>,
        batch_size: usize,
    ) -> Node {
        let mut root = if let Some(mut r) = root {
            let target_total = self.num_simulations;
            let already_done = r.visit_count;
            let nb_simulations = if already_done >= target_total {
                self.min_simulations
            } else {
                std::cmp::max(self.min_simulations, target_total - already_done)
            };

            r.prior = 0.0;
            r.update_depth(0);
            if !r.expanded() {
                let results = agent.predict(&[state]);
                let (action_probs, _) = results.into_iter().next().unwrap();
                r.expand(state.clone(), action_probs, true);
            }
            (r, nb_simulations)
        } else {
            let nb_simulations = self.num_simulations;
            let mut r = Node::new(0.0, depth);
            let results = agent.predict(&[state]);
            let (action_probs, _) = results.into_iter().next().unwrap();
            r.expand(state.clone(), action_probs, true);
            (r, nb_simulations)
        };

        let pb = if self.show_progress {
            let pb = ProgressBar::new(root.1 as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{msg} [{bar:40.cyan/blue}] {pos}/{len}")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            pb.set_message("MCTS Simulations");
            Some(pb)
        } else {
            None
        };

        let num_batches = (root.1 + batch_size - 1) / batch_size;

        for batch_idx in 0..num_batches {
            let current_batch_size =
                std::cmp::min(batch_size, root.1 - batch_idx * batch_size);

            let mut paths: Vec<Vec<Move>> = Vec::with_capacity(current_batch_size);
            let mut leaf_states: Vec<Board> = Vec::with_capacity(current_batch_size);
            let mut leaf_is_terminal: Vec<bool> = Vec::with_capacity(current_batch_size);
            let mut leaf_terminal_values: Vec<f64> = Vec::with_capacity(current_batch_size);
            let mut leaf_needs_expand: Vec<bool> = Vec::with_capacity(current_batch_size);

            for _ in 0..current_batch_size {
                let mut current_state = root.0.state.clone().unwrap();
                let mut search_path: Vec<Move> = vec![];
                let mut node_ref = &mut root.0 as *mut Node;

                unsafe {
                    while !current_state.is_terminal()
                        && (*node_ref).depth < self.temp_threshold
                    {
                        if !(*node_ref).expanded() {
                            break;
                        }

                        let (action, child) = (*node_ref).select_child();
                        search_path.push(action.clone());
                        current_state = current_state.apply_move(&action);
                        node_ref = child as *mut Node;

                        (*node_ref).visit_count += VIRTUAL_LOSS_COUNT;
                        (*node_ref).value_sum += VIRTUAL_LOSS_VALUE;
                    }

                    let mut vl_ref = &mut root.0 as *mut Node;
                    (*vl_ref).visit_count += VIRTUAL_LOSS_COUNT;
                    (*vl_ref).value_sum += VIRTUAL_LOSS_VALUE;
                    for action in &search_path[..search_path.len().saturating_sub(1)] {
                        if let Some(child) = (*vl_ref).children.get_mut(action) {
                            vl_ref = child as *mut Node;
                        }
                    }

                    let is_terminal = current_state.is_terminal();
                    let at_threshold = (*node_ref).depth >= self.temp_threshold;
                    let needs_expand = !is_terminal && !at_threshold && !(*node_ref).expanded();

                    let terminal_value = if is_terminal {
                        let winner = current_state.winner();
                        match winner {
                            None => 0.0,
                            Some(w) if w == current_state.current_player => 1.0,
                            _ => -1.0,
                        }
                    } else if at_threshold {
                        0.0
                    } else {
                        0.0
                    };

                    leaf_is_terminal.push(is_terminal || at_threshold);
                    leaf_terminal_values.push(terminal_value);
                    leaf_needs_expand.push(needs_expand);
                }

                paths.push(search_path);
                leaf_states.push(current_state);
            }

            let nn_indices: Vec<usize> = (0..current_batch_size)
                .filter(|&i| !leaf_is_terminal[i])
                .collect();

            let mut nn_results: Vec<(HashMap<Move, f32>, f32)> = Vec::new();
            if !nn_indices.is_empty() {
                let nn_states: Vec<&Board> =
                    nn_indices.iter().map(|&i| &leaf_states[i]).collect();
                nn_results = agent.predict(&nn_states);
            }

            let mut nn_result_idx = 0;
            for path_idx in 0..current_batch_size {
                let search_path = &paths[path_idx];

                let (value, action_probs) = if leaf_is_terminal[path_idx] {
                    (leaf_terminal_values[path_idx], HashMap::new())
                } else {
                    let (ap, v) = nn_results[nn_result_idx].clone();
                    nn_result_idx += 1;
                    (v as f64, ap)
                };

                unsafe {
                    let mut node_ref = &mut root.0 as *mut Node;
                    for action in search_path {
                        if let Some(child) = (*node_ref).children.get_mut(action) {
                            node_ref = child as *mut Node;
                        }
                    }

                    if leaf_needs_expand[path_idx] && !(*node_ref).expanded() {
                        (*node_ref)
                            .expand(leaf_states[path_idx].clone(), action_probs, false);
                    }

                    (*node_ref).visit_count -= VIRTUAL_LOSS_COUNT;
                    (*node_ref).value_sum -= VIRTUAL_LOSS_VALUE;
                    (*node_ref).value_sum += value;
                    (*node_ref).visit_count += 1;
                    let mut current_value = -value;

                    for action in search_path.iter().rev() {
                        let mut parent_ref = &mut root.0 as *mut Node;
                        for prev_action in search_path.iter() {
                            if prev_action == action {
                                break;
                            }
                            if let Some(child) = (*parent_ref).children.get_mut(prev_action) {
                                parent_ref = child as *mut Node;
                            }
                        }

                        (*parent_ref).value_sum += current_value;
                        (*parent_ref).visit_count += 1;
                        current_value = -current_value;
                    }

                    root.0.visit_count -= VIRTUAL_LOSS_COUNT;
                    root.0.value_sum -= VIRTUAL_LOSS_VALUE;
                    root.0.value_sum += current_value;
                    root.0.visit_count += 1;
                }
            }

            if let Some(ref pb) = pb {
                pb.inc(current_batch_size as u64);
            }
        }

        if let Some(pb) = pb {
            pb.finish_with_message("MCTS completed");
        }

        root.0
    }
}
