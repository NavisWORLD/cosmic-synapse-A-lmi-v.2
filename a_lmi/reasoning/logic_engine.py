"""
Logical Debate and Argumentation Framework

Implements structured argumentation, fallacy detection, and
counter-argument generation for reasoned discourse.
"""

import logging
from typing import Dict, Any, List, Tuple
import re


class LogicEngine:
    """
    Logical reasoning and argumentation engine.
    
    Features:
    - Argument structure analysis
    - Premise extraction
    - Fallacy detection
    - Counter-argument generation
    - Toulmin model implementation
    - Socratic questioning
    """
    
    def __init__(self):
        """Initialize logic engine."""
        self.logger = logging.getLogger(__name__)
        
        # Common logical fallacies
        self.fallacies = {
            'ad_hominem': r'(you|they|him|her)\s+(are|is)\s+(stupid|wrong|bad|crazy)',
            'strawman': r'(misrepresented|distorted|twisting)\s+(my|your|his|her)\s+(words|position)',
            'false_dilemma': r'(either|or|must|only)\s+(this|that)\s+(or)',
            'slippery_slope': r'(if|then|will)\s+(inevitably|certainly)\s+(lead to)',
            'appeal_to_emotion': r'(patriotic|loyal|for|trust|feeling|emotion)',
            'circular_reasoning': r'(\w+)\s+(is|are)\s+(true|correct)\s+because\s+\1',
            'hasty_generalization': r'(all|every|none|never|always)\s+(\w+)',
            'post_hoc': r'after\s+(that|this)\s+therefore\s+because',
            'red_herring': r'(but|however)\s+(what|why)\s+(about)'
        }
        
        self.logger.info("Logic engine initialized")
    
    def extract_premises(self, argument: str) -> List[str]:
        """
        Extract premises from an argument.
        
        Args:
            argument: Argument text
            
        Returns:
            List of premises
        """
        premises = []
        
        # Look for premise indicators
        premise_patterns = [
            r'(because|since|given that|as|for)\s+([^.!?]+)',
            r'([^.!?]+)\s+(therefore|hence|thus|so)',
            r'(premise|assume|suppose|given)\s+that\s+([^.!?]+)'
        ]
        
        for pattern in premise_patterns:
            matches = re.finditer(pattern, argument, re.IGNORECASE)
            for match in matches:
                premise = match.group(0).strip()
                premises.append(premise)
        
        return premises if premises else [argument]
    
    def extract_conclusion(self, argument: str) -> str:
        """
        Extract conclusion from argument.
        
        Args:
            argument: Argument text
            
        Returns:
            Conclusion statement
        """
        conclusion_indicators = [
            r'(therefore|hence|thus|so|consequently|it follows that)\s+([^.!?]+)',
            r'(conclusion|therefore we can|prove that)\s+([^.!?]+)'
        ]
        
        for pattern in conclusion_indicators:
            match = re.search(pattern, argument, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        
        # If no explicit conclusion indicator, assume last sentence
        sentences = argument.split('.')
        return sentences[-1].strip() if sentences else argument
    
    def detect_fallacies(self, argument: str) -> List[Dict[str, Any]]:
        """
        Detect logical fallacies in argument.
        
        Args:
            argument: Argument to analyze
            
        Returns:
            List of detected fallacies with evidence
        """
        detected = []
        
        for fallacy_name, pattern in self.fallacies.items():
            matches = re.finditer(pattern, argument, re.IGNORECASE)
            for match in matches:
                detected.append({
                    'fallacy': fallacy_name,
                    'evidence': match.group(0),
                    'position': match.start(),
                    'description': self._get_fallacy_description(fallacy_name)
                })
        
        self.logger.info(f"Detected {len(detected)} potential fallacies")
        return detected
    
    def _get_fallacy_description(self, fallacy_name: str) -> str:
        """Get human-readable description of fallacy."""
        descriptions = {
            'ad_hominem': 'Personal attack instead of addressing the argument',
            'strawman': 'Misrepresenting opponent\'s position',
            'false_dilemma': 'Presenting only two options when more exist',
            'slippery_slope': 'Assuming one action will lead to extreme consequence',
            'appeal_to_emotion': 'Relying on emotion rather than logic',
            'circular_reasoning': 'Conclusion is assumed in premises',
            'hasty_generalization': 'Jumping to conclusions from limited evidence',
            'post_hoc': 'Assuming causation from correlation',
            'red_herring': 'Introducing irrelevant topic to distract'
        }
        return descriptions.get(fallacy_name, 'Unknown fallacy')
    
    def build_argument_structure(self, argument: str) -> Dict[str, Any]:
        """
        Analyze argument structure using Toulmin model.
        
        Args:
            argument: Argument text
            
        Returns:
            Structured argument with claim, grounds, warrant, etc.
        """
        premises = self.extract_premises(argument)
        conclusion = self.extract_conclusion(argument)
        fallacies = self.detect_fallacies(argument)
        
        # Identify warrant (connection between premises and conclusion)
        warrant = self._identify_warrant(premises, conclusion)
        
        return {
            'claim': conclusion,
            'grounds': premises,
            'warrant': warrant,
            'fallacies': fallacies,
            'strength': self._assess_argument_strength(premises, fallacies)
        }
    
    def _identify_warrant(self, premises: List[str], conclusion: str) -> str:
        """Identify the warrant connecting premises to conclusion."""
        # Simplified: look for logical connectors
        if 'if' in premises[0].lower() or 'when' in premises[0].lower():
            return 'Conditional reasoning'
        elif 'all' in premises[0].lower() or 'every' in premises[0].lower():
            return 'Universal generalization'
        else:
            return 'Implied logical connection'
    
    def _assess_argument_strength(self, premises: List[str], fallacies: List[Dict]) -> float:
        """Assess argument strength on 0-1 scale."""
        strength = 1.0
        
        # Reduce strength for each fallacy
        strength -= len(fallacies) * 0.15
        
        # Reward multiple premises
        if len(premises) > 1:
            strength += 0.1
        
        return max(0.0, min(1.0, strength))
    
    def generate_counter_argument(self, argument: str) -> Dict[str, Any]:
        """
        Generate counter-argument to provided argument.
        
        Args:
            argument: Original argument
            
        Returns:
            Structured counter-argument
        """
        structure = self.build_argument_structure(argument)
        
        # Generate counter based on detected fallacies
        if structure['fallacies']:
            counter_strategy = f"Challenge the logical fallacy: {structure['fallacies'][0]['fallacy']}"
        else:
            counter_strategy = "Present alternative evidence or perspective"
        
        # Build counter argument
        counter_premises = []
        
        for premise in structure['grounds']:
            # Negate or question each premise
            counter_premises.append(f"However, {premise} may not hold because...")
        
        return {
            'original_claim': structure['claim'],
            'counter_claim': f"On the contrary, {structure['claim'].lower()} may not be valid",
            'counter_premises': counter_premises,
            'strategy': counter_strategy,
            'strength': 0.6  # Default counter strength
        }
    
    def socratic_questioning(self, statement: str) -> List[str]:
        """
        Generate Socratic questions to examine statement.
        
        Args:
            statement: Statement to question
            
        Returns:
            List of probing questions
        """
        questions = [
            f"What do you mean by \"{statement}\"?",
            f"What evidence supports this claim?",
            f"How does this relate to what we already know?",
            f"What alternative explanations might exist?",
            f"What are the implications if this is true?",
            f"What would we expect to observe if this were false?"
        ]
        
        return questions

