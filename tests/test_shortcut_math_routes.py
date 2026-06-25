from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / 'first_principles_ai'


class ShortcutMathRouteTests(unittest.TestCase):
    def test_main_cli_does_not_expose_direct_text_solver_routes(self):
        main_text = (PACKAGE_ROOT / 'main.py').read_text(encoding='utf-8')

        forbidden_fragments = [
            '--ground-up-math-problems',
            '--math-solver',
            '--unified-science-math-system',
            'run_math_problem_benchmark',
            'run_symbolic_math_solver',
            'expected_answers',
        ]
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, main_text)

    def test_answer_key_problem_bank_modules_are_not_present(self):
        forbidden_paths = [
            PACKAGE_ROOT / 'math_solver.py',
            PACKAGE_ROOT / 'math_problem_benchmark.py',
            PACKAGE_ROOT / 'math_sources.py',
            PACKAGE_ROOT / 'math_science_curriculum.py',
            PACKAGE_ROOT / 'data' / 'math_source_registry.json',
            PACKAGE_ROOT / 'data' / 'imo2024_shortlist_open_problems.json',
        ]

        for path in forbidden_paths:
            self.assertFalse(path.exists(), f'{path} should not be in the discovery package')


if __name__ == '__main__':
    unittest.main()
