use std::collections::HashMap;
use std::hash::Hash;

use pyo3::prelude::*;

pub mod agent;
pub mod game;
pub mod search;
pub mod training;

use game::{Board, Move};
use search::mcts::{Node, MCTS};

// Python wrapper for Board
#[pyclass]
#[derive(Clone)]
struct PyBoard {
    board: Board,
}

#[pymethods]
impl PyBoard {
    #[new]
    fn new() -> Self {
        PyBoard {
            board: Board::new(),
        }
    }

    fn legal_moves(&self) -> Vec<PyMove> {
        self.board
            .legal_moves()
            .into_iter()
            .map(|m| PyMove { mov: m })
            .collect()
    }

    fn apply_move(&self, mov: &PyMove) -> PyBoard {
        PyBoard {
            board: self.board.apply_move(&mov.mov),
        }
    }

    fn is_terminal(&self) -> bool {
        self.board.is_terminal()
    }

    fn winner(&self) -> Option<i32> {
        self.board.winner().map(|p| p as i32)
    }

    fn to_embed(&self) -> Vec<f32> {
        self.board.to_embed()
    }

    fn current_player(&self) -> i32 {
        self.board.current_player as i32
    }

    fn print_board(&self) {
        self.board.print_board()
    }
}

// Python wrapper for Move
#[pyclass]
#[derive(Clone, Hash, PartialEq, Eq)]
struct PyMove {
    mov: Move,
}

#[pymethods]
impl PyMove {
    #[new]
    #[pyo3(signature = (*, to_position, from_position=None, removed_position=None))]
    fn new(
        to_position: usize,
        from_position: Option<usize>,
        removed_position: Option<usize>,
    ) -> Self {
        PyMove {
            mov: Move {
                from_position,
                to_position,
                removed_position,
            },
        }
    }

    #[staticmethod]
    #[pyo3(name = "placement")]
    fn placement(position: usize) -> PyMove {
        PyMove {
            mov: Move::placement(position),
        }
    }

    #[staticmethod]
    #[pyo3(name = "move_piece")]
    fn move_piece(from_position: usize, to_position: usize) -> PyMove {
        PyMove {
            mov: Move::move_piece(from_position, to_position),
        }
    }

    fn with_removal(&self, removed_position: usize) -> PyMove {
        PyMove {
            mov: self.mov.with_removal(removed_position),
        }
    }

    fn to_embed(&self) -> Vec<f32> {
        self.mov.to_embed()
    }

    fn to_indices(&self) -> Vec<i64> {
        self.mov.to_indices()
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_position(&self) -> Option<usize> {
        self.mov.from_position
    }

    fn to_position(&self) -> usize {
        self.mov.to_position
    }

    fn removed_position(&self) -> Option<usize> {
        self.mov.removed_position
    }

    fn __repr__(&self) -> String {
        format!(
            "Move(from={:?}, to={}, removed={:?})",
            self.mov.from_position, self.mov.to_position, self.mov.removed_position
        )
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.mov.hash(&mut hasher);
        hasher.finish()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.mov == other.mov
    }

    fn __ne__(&self, other: &Self) -> bool {
        self.mov != other.mov
    }
}

// Python wrapper for MCTS
#[pyclass]
struct PyMCTS {
    mcts: MCTS,
}

#[pymethods]
impl PyMCTS {
    #[new]
    #[pyo3(signature = (num_simulations, show_progress=false))]
    fn new(num_simulations: usize, show_progress: bool) -> Self {
        PyMCTS {
            mcts: MCTS::new(num_simulations, show_progress),
        }
    }

    #[pyo3(signature = (agent, board, depth, root=None))]
    fn run(
        &self,
        agent: &Bound<'_, PyAny>,
        board: &PyBoard,
        depth: usize,
        root: Option<PyNode>,
    ) -> PyNode {
        let rust_agent = agent::base::PythonAgent::new(agent.clone().unbind());
        let rust_root = root.map(|n| n.node);
        let result_node = self.mcts.run(&rust_agent, &board.board, depth, rust_root);
        PyNode { node: result_node }
    }
}

// Python wrapper for Node
#[pyclass]
#[derive(Clone)]
struct PyNode {
    node: Node,
}

#[pymethods]
impl PyNode {
    #[getter]
    fn value(&self) -> f64 {
        self.node.value()
    }

    #[getter]
    fn visit_count(&self) -> usize {
        self.node.visit_count
    }

    #[getter]
    fn children(&self) -> HashMap<PyMove, PyNode> {
        self.node
            .children
            .iter()
            .map(|(mov, node)| (PyMove { mov: mov.clone() }, PyNode { node: node.clone() }))
            .collect()
    }

    #[getter]
    fn prior(&self) -> f64 {
        self.node.prior
    }

    fn select_action(&self, temperature: f64) -> PyMove {
        let action = self.node.select_action(temperature);
        PyMove { mov: action }
    }
}

// Python module
#[pymodule]
fn jdm_ru(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBoard>()?;
    m.add_class::<PyMove>()?;
    m.add_class::<PyMCTS>()?;
    m.add_class::<PyNode>()?;
    m.add_function(wrap_pyfunction!(
        training::self_play::generate_train_examples,
        m
    )?)?;
    Ok(())
}
