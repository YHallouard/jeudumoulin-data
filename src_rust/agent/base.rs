use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::HashMap;

use crate::game::{Board, Move};

pub trait Agent {
    fn predict(&self, states: &[&Board]) -> Vec<(HashMap<Move, f32>, f32)>;
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
    fn predict(&self, states: &[&Board]) -> Vec<(HashMap<Move, f32>, f32)> {
        Python::with_gil(|py| {
            let state_embeddings: Vec<Vec<f32>> = states.iter().map(|s| s.to_embed()).collect();

            let all_legal_moves: Vec<Vec<Move>> =
                states.iter().map(|s| s.legal_moves()).collect();

            let legal_moves_as_lists: Vec<Vec<Vec<Option<i64>>>> = all_legal_moves
                .iter()
                .map(|moves| {
                    moves
                        .iter()
                        .map(|m| {
                            vec![
                                m.from_position.map(|p| p as i64),
                                Some(m.to_position as i64),
                                m.removed_position.map(|p| p as i64),
                            ]
                        })
                        .collect()
                })
                .collect();

            let py_embeddings = PyList::new_bound(
                py,
                state_embeddings
                    .iter()
                    .map(|emb| PyList::new_bound(py, emb)),
            );
            let py_legal_moves = PyList::new_bound(
                py,
                legal_moves_as_lists.iter().map(|moves| {
                    PyList::new_bound(
                        py,
                        moves.iter().map(|m| PyList::new_bound(py, m)),
                    )
                }),
            );

            let result = self
                .agent
                .call_method1(py, "predict", (py_embeddings, py_legal_moves))
                .expect("Failed to call predict");

            let results: Vec<(HashMap<usize, f32>, f32)> = result
                .extract(py)
                .expect("Failed to extract batch prediction result");

            results
                .into_iter()
                .enumerate()
                .map(|(i, (policy_dict, value))| {
                    let legal_moves = &all_legal_moves[i];
                    let mut action_probs = HashMap::new();
                    for (idx, prob) in policy_dict {
                        if idx < legal_moves.len() {
                            action_probs.insert(legal_moves[idx].clone(), prob);
                        }
                    }
                    (action_probs, value)
                })
                .collect()
        })
    }
}
