use std::collections::HashMap;

use indicatif::{ProgressBar, ProgressStyle};
use rand::distributions::WeightedIndex;
use rand::prelude::*;

use crate::agent::base::Agent;
use crate::game::{Board, Move};

const C_PUCT: f64 = 1.25; //std::f64::consts::SQRT_2;

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

    pub fn run(&self, agent: &dyn Agent, state: &Board, depth: usize, root: Option<Node>) -> Node {
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
                let (action_probs, _) = agent.predict(state);
                r.expand(state.clone(), action_probs, true);
            }
            (r, nb_simulations)
        } else {
            let nb_simulations = self.num_simulations;
            let mut r = Node::new(0.0, depth);
            let (action_probs, _) = agent.predict(state);
            r.expand(state.clone(), action_probs, true);
            (r, nb_simulations)
        };

        // println!("MCTS: Starting {} simulations", root.1);

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
            // Select path through tree
            let mut current_state = root.0.state.clone().unwrap();
            let mut search_path: Vec<Move> = vec![];
            let mut node_ref = &mut root.0 as *mut Node;

            unsafe {
                // Navigate down the tree
                while !current_state.is_terminal() && (*node_ref).depth < self.temp_threshold {
                    if !(*node_ref).expanded() {
                        break;
                    }

                    let (action, child) = (*node_ref).select_child();
                    search_path.push(action.clone());
                    current_state = current_state.apply_move(&action);
                    node_ref = child as *mut Node;
                }

                // Evaluate leaf node
                let (value, action_probs) = if current_state.is_terminal() {
                    let winner = current_state.winner();
                    let value = match winner {
                        None => 0.0,
                        Some(w) if w == current_state.current_player => 1.0,
                        _ => -1.0,
                    };
                    (value, HashMap::new())
                } else if (*node_ref).depth < self.temp_threshold {
                    let (action_probs, value) = agent.predict(&current_state);
                    (value as f64, action_probs)
                } else {
                    (0.0, HashMap::new())
                };

                // Expand leaf node
                if !current_state.is_terminal() && (*node_ref).depth < self.temp_threshold {
                    (*node_ref).expand(current_state.clone(), action_probs, false);
                }

                // Backpropagate
                let current_node = node_ref;
                let mut current_value = value;

                // Update leaf
                (*current_node).value_sum += current_value;
                (*current_node).visit_count += 1;
                current_value = -current_value;

                // Walk back up the tree
                for action in search_path.iter().rev() {
                    // Find parent
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

                // Update root
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
}
