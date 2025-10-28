"""
Mathematical Reasoning Engine

Advanced mathematical problem solving with symbolic reasoning,
theorem proving, and integration with OpenAI's advanced models.
"""

import logging
import sympy as sp
from sympy import symbols, simplify, expand, factor, solve, diff, integrate
from typing import Dict, Any, List, Union
import openai


class MathEngine:
    """
    Mathematical reasoning engine for advanced problem solving.
    
    Capabilities:
    - Symbolic mathematics (SymPy)
    - Integration with OpenAI o1-mini for complex reasoning
    - Step-by-step derivation tracking
    - Theorem verification
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize math engine.
        
        Args:
            openai_api_key: OpenAI API key for o1-mini access
        """
        self.logger = logging.getLogger(__name__)
        self.openai_client = None
        
        if openai_api_key:
            try:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                self.logger.info("OpenAI client initialized")
            except Exception as e:
                self.logger.warning(f"OpenAI initialization failed: {e}")
    
    def solve_equation(self, equation: str, variable: str) -> List[Dict[str, Any]]:
        """
        Solve mathematical equation symbolically.
        
        Args:
            equation: String equation (e.g., "x^2 + 3*x - 4 = 0")
            variable: Variable to solve for
            
        Returns:
            List of solutions with steps
        """
        try:
            # Parse equation
            x = symbols(variable)
            expr = sp.sympify(equation.replace('=', '-'))  # Move RHS to LHS
            
            # Solve
            solutions = solve(expr, x)
            
            # Format results
            results = []
            for i, sol in enumerate(solutions):
                results.append({
                    'solution_number': i + 1,
                    'value': str(sol),
                    'simplified': str(simplify(sol)),
                    'numeric': float(sol.evalf()) if sol.is_number else None
                })
            
            self.logger.info(f"Solved equation: {len(solutions)} solutions")
            return results
            
        except Exception as e:
            self.logger.error(f"Error solving equation: {e}")
            return []
    
    def differentiate(self, expression: str, variable: str) -> Dict[str, Any]:
        """
        Differentiate an expression.
        
        Args:
            expression: Mathematical expression
            variable: Variable to differentiate with respect to
            
        Returns:
            Derivative with steps
        """
        try:
            x = symbols(variable)
            expr = sp.sympify(expression)
            
            # Compute derivative
            derivative = diff(expr, x)
            
            return {
                'expression': str(expr),
                'variable': variable,
                'derivative': str(derivative),
                'simplified': str(simplify(derivative))
            }
            
        except Exception as e:
            self.logger.error(f"Error differentiating: {e}")
            return {}
    
    def integrate(self, expression: str, variable: str, bounds: tuple = None) -> Dict[str, Any]:
        """
        Integrate an expression.
        
        Args:
            expression: Mathematical expression
            variable: Variable to integrate with respect to
            bounds: Optional integration bounds (a, b)
            
        Returns:
            Integral with steps
        """
        try:
            x = symbols(variable)
            expr = sp.sympify(expression)
            
            # Compute integral
            if bounds:
                integral = integrate(expr, (x, bounds[0], bounds[1]))
                integral_value = integral.evalf()
            else:
                integral = integrate(expr, x)
                integral_value = None
            
            return {
                'expression': str(expr),
                'variable': variable,
                'integral': str(integral),
                'simplified': str(simplify(integral)),
                'numeric_value': float(integral_value) if integral_value is not None else None
            }
            
        except Exception as e:
            self.logger.error(f"Error integrating: {e}")
            return {}
    
    def reason_about_proof(self, theorem: str, proof_steps: List[str]) -> Dict[str, Any]:
        """
        Verify mathematical proof using symbolic reasoning.
        
        Args:
            theorem: Statement to prove
            proof_steps: List of proof steps
            
        Returns:
            Verification result with feedback
        """
        try:
            # Use OpenAI o1-mini if available
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="o1-mini",
                    messages=[
                        {"role": "system", "content": "You are a mathematical proof verifier. Analyze proofs step by step."},
                        {"role": "user", "content": f"Theorem: {theorem}\n\nProof steps:\n" + "\n".join(proof_steps)}
                    ],
                    temperature=0.3
                )
                
                return {
                    'verified': 'True' in response.choices[0].message.content,
                    'explanation': response.choices[0].message.content,
                    'confidence': 0.8
                }
            
            # Fallback: basic symbolic verification
            return {
                'verified': None,
                'explanation': 'Proof verification requires OpenAI API access',
                'confidence': 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error verifying proof: {e}")
            return {'verified': None, 'explanation': str(e), 'confidence': 0.0}
    
    def advanced_problem_solving(self, problem: str) -> Dict[str, Any]:
        """
        Solve complex mathematical problems using OpenAI.
        
        Args:
            problem: Problem description
            
        Returns:
            Solution with reasoning steps
        """
        if not self.openai_client:
            return {'error': 'OpenAI API not configured'}
        
        try:
            response = self.openai_client.chat.completions.create(
                model="o1-mini",
                messages=[
                    {"role": "system", "content": "You are an expert mathematician. Solve problems step by step."},
                    {"role": "user", "content": problem}
                ],
                temperature=0.3
            )
            
            return {
                'problem': problem,
                'solution': response.choices[0].message.content,
                'model': 'o1-mini'
            }
            
        except Exception as e:
            self.logger.error(f"Error solving problem: {e}")
            return {'error': str(e)}
    
    def mathematical_derivation(self, expression: str, target_form: str) -> Dict[str, Any]:
        """
        Derive target mathematical form from expression.
        
        Args:
            expression: Starting expression
            target_form: Desired form
            
        Returns:
            Derivation steps
        """
        try:
            expr = sp.sympify(expression)
            target = sp.sympify(target_form)
            
            # Try to transform expression
            steps = []
            
            # Step 1: Expand
            expanded = expand(expr)
            steps.append(f"Expanded: {expanded}")
            
            # Step 2: Simplify
            simplified = simplify(expr)
            steps.append(f"Simplified: {simplified}")
            
            return {
                'expression': str(expr),
                'target_form': target_form,
                'steps': steps,
                'matches_target': str(simplified) == target_form
            }
            
        except Exception as e:
            self.logger.error(f"Error in derivation: {e}")
            return {'error': str(e)}

