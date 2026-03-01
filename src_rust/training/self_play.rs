use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rand::Rng;
use std::collections::HashMap;

use indicatif::{ProgressBar, ProgressStyle};

use crate::game::{Board, Move, Player};
use crate::search::{Node, MCTS};

type TrainExamplesResult = (
    Vec<Vec<f32>>,
    Vec<Vec<Vec<Option<usize>>>>,
    Vec<Vec<f32>>,
    Vec<f32>,
);

pub struct TrainExample {
    pub state_embedding: Vec<f32>,
    pub legal_moves: Vec<Vec<Option<usize>>>,
    pub policy_labels: Vec<f32>,
    pub value_label: f32,
}

struct PythonAgent {
    agent: PyObject,
}

impl PythonAgent {
    fn new(agent: PyObject) -> Self {
        PythonAgent { agent }
    }
}

impl crate::agent::base::Agent for PythonAgent {
    fn predict(&self, state: &Board) -> (HashMap<Move, f32>, f32) {
        Python::with_gil(|py| {
            let state_embedding = state.to_embed();
            let legal_moves = state.legal_moves();

            let legal_moves_as_lists: Vec<Vec<Option<usize>>> = legal_moves
                .iter()
                .map(|m| vec![m.from_position, Some(m.to_position), m.removed_position])
                .collect();

            let py_state_embedding = PyList::new_bound(py, &state_embedding);
            let py_legal_moves = PyList::new_bound(
                py,
                legal_moves_as_lists
                    .iter()
                    .map(|m| PyList::new_bound(py, m)),
            );

            let result = self
                .agent
                .call_method1(py, "predict", (py_state_embedding, py_legal_moves))
                .expect("Failed to call predict");

            let result_tuple: &Bound<PyTuple> = result
                .downcast_bound::<PyTuple>(py)
                .expect("Result is not a tuple");

            let policy_item = result_tuple.get_item(0).expect("No policy in result");
            let policy_dict: &Bound<PyDict> = policy_item
                .downcast::<PyDict>()
                .expect("Policy is not a dict");

            let value_item = result_tuple.get_item(1).expect("No value in result");
            let value: f32 = value_item.extract().expect("Value is not a float");

            let mut action_probs: HashMap<Move, f32> = HashMap::new();
            for (key, val) in policy_dict.iter() {
                let move_idx: usize = key.extract().expect("Key is not usize");
                let prob: f32 = val.extract().expect("Value is not f32");
                if move_idx < legal_moves.len() {
                    action_probs.insert(legal_moves[move_idx].clone(), prob);
                }
            }

            (action_probs, value)
        })
    }
}

pub fn execute_episode(
    py: Python,
    agent: &PyObject,
    num_simulations: usize,
    max_episode_steps: usize,
    temperature: f64,
    pb: &ProgressBar,
) -> PyResult<Vec<TrainExample>> {
    let mut train_examples: Vec<TrainExample> = Vec::new();
    let mut state = Board::new();
    let mut episode_step = 0;

    let python_agent = PythonAgent::new(agent.clone_ref(py));
    let mcts = MCTS::new(num_simulations, false);
    let mut reused_root: Option<Node> = None;

    while episode_step < max_episode_steps && !state.is_terminal() {
        episode_step += 1;
        pb.set_message(format!("Step {}/{}", episode_step, max_episode_steps));

        let root = mcts.run(&python_agent, &state, 0, reused_root);

        let _state_embedding = state.to_embed();
        let legal_moves = state.legal_moves();

        let _legal_moves_as_lists: Vec<Vec<Option<usize>>> = legal_moves
            .iter()
            .map(|m| vec![m.from_position, Some(m.to_position), m.removed_position])
            .collect();

        let mut action_probs: HashMap<Move, f32> =
            legal_moves.iter().map(|m| (m.clone(), 0.0)).collect();

        let total_count: usize = root.children.values().map(|n| n.visit_count).sum();
        if total_count > 0 {
            for (action, child) in &root.children {
                let prob = child.visit_count as f32 / total_count as f32;
                action_probs.insert(action.clone(), prob);
            }
        }

        let _policy_labels: Vec<f32> = legal_moves
            .iter()
            .map(|m| *action_probs.get(m).unwrap_or(&0.0))
            .collect();

        let random_rotation = rand::thread_rng().gen_range(0..4);
        let rotated_state = state.rotate(random_rotation);
        let rotated_state_embedding = rotated_state.to_embed();

        let rotated_legal_moves: Vec<Move> = legal_moves
            .iter()
            .map(|m| m.rotate(random_rotation))
            .collect();

        let rotated_legal_moves_as_lists: Vec<Vec<Option<usize>>> = rotated_legal_moves
            .iter()
            .map(|m| vec![m.from_position, Some(m.to_position), m.removed_position])
            .collect();

        let rotated_action_probs: HashMap<Move, f32> = action_probs
            .iter()
            .map(|(m, prob)| (m.rotate(random_rotation), *prob))
            .collect();

        let rotated_policy_labels: Vec<f32> = rotated_legal_moves
            .iter()
            .map(|m| *rotated_action_probs.get(m).unwrap_or(&0.0))
            .collect();

        train_examples.push(TrainExample {
            state_embedding: rotated_state_embedding,
            legal_moves: rotated_legal_moves_as_lists,
            policy_labels: rotated_policy_labels,
            value_label: 0.0,
        });

        let action = root.select_action(temperature);
        reused_root = root
            .children
            .into_iter()
            .find(|(k, _)| k == &action)
            .map(|(_, v)| v);

        state = state.apply_move(&action);
    }

    if state.is_terminal() {
        let winner = state.winner();
        let final_player = state.current_player;

        let reward = match winner {
            None => 0.0,
            Some(w) if w == final_player => 1.0,
            _ => -1.0,
        };

        for example in &mut train_examples {
            let example_player = if example.state_embedding[0] == 1.0 {
                Player::White
            } else {
                Player::Black
            };

            let value_multiplier = if example_player != final_player {
                -1.0
            } else {
                1.0
            };

            example.value_label = reward * value_multiplier;
        }
    }

    Ok(train_examples)
}

#[pyfunction]
pub fn generate_train_examples(
    py: Python,
    agent: PyObject,
    num_simulations: usize,
    num_episodes: usize,
    max_episode_steps: usize,
    temperature: f64,
) -> PyResult<TrainExamplesResult> {
    let mut all_state_embeddings: Vec<Vec<f32>> = Vec::new();
    let mut all_legal_moves: Vec<Vec<Vec<Option<usize>>>> = Vec::new();
    let mut all_policy_labels: Vec<Vec<f32>> = Vec::new();
    let mut all_value_labels: Vec<f32> = Vec::new();

    let pb = ProgressBar::new(num_episodes as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} [{elapsed_precise}]")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message("Self-play episodes");

    for episode_idx in 0..num_episodes {
        pb.set_message(format!("Episode {}/{}", episode_idx + 1, num_episodes));

        let examples = execute_episode(
            py,
            &agent,
            num_simulations,
            max_episode_steps,
            temperature,
            &pb,
        )?;

        for example in examples {
            all_state_embeddings.push(example.state_embedding);
            all_legal_moves.push(example.legal_moves);
            all_policy_labels.push(example.policy_labels);
            all_value_labels.push(example.value_label);
        }

        pb.inc(1);
    }

    pb.finish_with_message("Self-play complete");

    Ok((
        all_state_embeddings,
        all_legal_moves,
        all_policy_labels,
        all_value_labels,
    ))
}
