#[cfg(test)]
mod test_self_play {
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};

    #[test]
    fn test_generate_train_examples_returns_correct_types() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let agent_code = r#"
class SimpleAgent:
    def predict(self, state_embedding, legal_moves):
        num_moves = len(legal_moves)
        policy = {i: 1.0 / num_moves for i in range(num_moves)}
        value = 0.0
        return policy, value

agent = SimpleAgent()
"#;

            py.run_bound(agent_code, None, None).unwrap();
            let agent = py.eval_bound("agent", None, None).unwrap();

            let result = crate::training::self_play::generate_train_examples(
                py,
                agent.into(),
                5,
                1,
                10,
                1.0,
            );

            assert!(result.is_ok());
            let (states, legal_moves, policies, values) = result.unwrap();
            assert!(!states.is_empty());
            assert_eq!(states.len(), legal_moves.len());
            assert_eq!(states.len(), policies.len());
            assert_eq!(states.len(), values.len());
        });
    }
}
