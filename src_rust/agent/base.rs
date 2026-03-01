use pyo3::prelude::*;
use std::collections::HashMap;

use crate::game::{Board, Move};

pub trait Agent {
    fn predict(&self, state: &Board) -> (HashMap<Move, f32>, f32);
}

pub struct PythonAgent {
    agent: PyObject,
}

impl PythonAgent {
    pub fn new(agent: PyObject) -> Self {
        PythonAgent { agent }
    }
}

impl Agent for PythonAgent {
    fn predict(&self, state: &Board) -> (HashMap<Move, f32>, f32) {
        Python::with_gil(|py| {
            let state_embedding = state.to_embed();
            let legal_moves = state.legal_moves();
            let legal_moves_as_lists: Vec<Vec<Option<i64>>> = legal_moves
                .iter()
                .map(|m| {
                    vec![
                        m.from_position.map(|p| p as i64),
                        Some(m.to_position as i64),
                        m.removed_position.map(|p| p as i64),
                    ]
                })
                .collect();

            let result = self
                .agent
                .call_method1(py, "predict", (state_embedding, legal_moves_as_lists))
                .expect("Failed to call predict");

            let (policy_dict, value): (HashMap<usize, f32>, f32) = result
                .extract(py)
                .expect("Failed to extract prediction result");

            let mut action_probs = HashMap::new();
            for (idx, prob) in policy_dict {
                if idx < legal_moves.len() {
                    action_probs.insert(legal_moves[idx].clone(), prob);
                }
            }

            (action_probs, value)
        })
    }
}
