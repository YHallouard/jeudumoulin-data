use pyo3::prelude::*;
use rand::Rng;
use std::collections::HashMap;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};

use crate::agent::base::PythonAgent;
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

pub fn execute_episode(
    py: Python,
    agent: &PyObject,
    num_simulations: usize,
    max_episode_steps: usize,
    temperature: f64,
    batch_size: usize,
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

        let root =
            mcts.run_batched(&python_agent, &state, 0, reused_root, batch_size);

        let legal_moves = state.legal_moves();

        let mut action_probs: HashMap<Move, f32> =
            legal_moves.iter().map(|m| (m.clone(), 0.0)).collect();

        let total_count: usize = root.children.values().map(|n| n.visit_count).sum();
        if total_count > 0 {
            for (action, child) in &root.children {
                let prob = child.visit_count as f32 / total_count as f32;
                action_probs.insert(action.clone(), prob);
            }
        }

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
#[pyo3(signature = (agent, num_simulations, num_episodes, max_episode_steps, temperature, batch_size=8))]
pub fn generate_train_examples(
    py: Python,
    agent: PyObject,
    num_simulations: usize,
    num_episodes: usize,
    max_episode_steps: usize,
    temperature: f64,
    batch_size: usize,
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

    let global_start = Instant::now();

    for episode_idx in 0..num_episodes {
        pb.set_message(format!("Episode {}/{}", episode_idx + 1, num_episodes));
        let episode_start = Instant::now();

        let examples = execute_episode(
            py,
            &agent,
            num_simulations,
            max_episode_steps,
            temperature,
            batch_size,
            &pb,
        )?;

        let num_examples = examples.len();
        for example in examples {
            all_state_embeddings.push(example.state_embedding);
            all_legal_moves.push(example.legal_moves);
            all_policy_labels.push(example.policy_labels);
            all_value_labels.push(example.value_label);
        }

        let episode_elapsed = episode_start.elapsed();
        let total_elapsed = global_start.elapsed();
        println!(
            "[self-play] episode {}/{} done in {:.1}s ({} steps) | total elapsed: {:.1}s",
            episode_idx + 1,
            num_episodes,
            episode_elapsed.as_secs_f64(),
            num_examples,
            total_elapsed.as_secs_f64(),
        );

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
