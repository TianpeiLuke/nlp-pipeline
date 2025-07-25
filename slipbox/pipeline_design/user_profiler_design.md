# User Profiler Design

## Overview

The User Profiler is a core component of the [Context Analysis Framework](./context_analysis_framework.md) that analyzes user characteristics to build comprehensive user profiles. It tracks expertise levels, preferences, success patterns, and learning progression to enable personalized configuration experiences.

## Purpose and Responsibilities

1. **Expertise Analysis**: Determine user expertise level across different domains
2. **Preference Tracking**: Learn and adapt to user interface and configuration preferences
3. **Success Pattern Recognition**: Identify patterns that lead to successful configurations
4. **Learning Velocity Assessment**: Track how quickly users adopt new features and complexity
5. **Profile Evolution**: Update profiles based on user activity and feedback

## Core Components

### 1. User Profiler (Main Component)

```python
class UserProfiler:
    """
    Analyzes user characteristics to build comprehensive user profiles
    
    Tracks expertise levels, preferences, success patterns, and learning progression
    """
    
    def __init__(self):
        self.profile_storage = UserProfileStorage()
        self.expertise_analyzer = ExpertiseAnalyzer()
        self.preference_tracker = PreferenceTracker()
        self.success_pattern_analyzer = SuccessPatternAnalyzer()
    
    def get_profile(self, user_id: str) -> UserProfile:
        """
        Get comprehensive user profile
        
        Args:
            user_id: User identifier
            
        Returns:
            UserProfile with expertise, preferences, and patterns
        """
        # Load existing profile or create new one
        existing_profile = self.profile_storage.load_profile(user_id)
        if not existing_profile:
            return self._create_initial_profile(user_id)
        
        # Update profile with recent activity
        updated_profile = self._update_profile_with_recent_activity(existing_profile)
        
        return updated_profile
    
    def update_profile_with_feedback(self,
                                   profile: UserProfile,
                                   feedback: ConfigurationFeedback) -> UserProfile:
        """
        Update user profile based on configuration feedback
        
        Args:
            profile: Current user profile
            feedback: Feedback from configuration usage
            
        Returns:
            Updated user profile
        """
        # Update expertise based on feedback
        updated_expertise = self.expertise_analyzer.update_expertise_with_feedback(
            profile.expertise_level, profile.domain_expertise, feedback
        )
        
        # Update preferences based on usage patterns
        updated_preferences = self.preference_tracker.update_preferences_with_feedback(
            profile.interface_preference, feedback
        )
        
        # Update success/failure patterns
        updated_patterns = self.success_pattern_analyzer.update_patterns_with_feedback(
            profile.success_patterns, profile.failure_patterns, feedback
        )
        
        return profile.with_updates(
            expertise_level=updated_expertise.level,
            domain_expertise=updated_expertise.domain_expertise,
            interface_preference=updated_preferences.interface_preference,
            success_patterns=updated_patterns.success_patterns,
            failure_patterns=updated_patterns.failure_patterns,
            learning_velocity=updated_expertise.learning_velocity,
            last_updated=datetime.now()
        )
    
    def _create_initial_profile(self, user_id: str) -> UserProfile:
        """Create initial profile for new user"""
        return UserProfile(
            user_id=user_id,
            expertise_level=UserExpertiseLevel.BEGINNER,
            interface_preference=InterfacePreference.GUIDED,
            domain_expertise={},
            success_patterns=[],
            failure_patterns=[],
            learning_velocity=LearningVelocity.MEDIUM,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
    
    def _update_profile_with_recent_activity(self, profile: UserProfile) -> UserProfile:
        """Update profile based on recent user activity"""
        # Analyze recent configurations
        recent_configs = self.profile_storage.get_recent_configurations(
            profile.user_id, days=30
        )
        
        # Update expertise level
        updated_expertise = self.expertise_analyzer.analyze_expertise(
            profile, recent_configs
        )
        
        # Update preferences
        updated_preferences = self.preference_tracker.analyze_preferences(
            profile, recent_configs
        )
        
        # Update success patterns
        updated_patterns = self.success_pattern_analyzer.analyze_patterns(
            profile, recent_configs
        )
        
        return profile.with_updates(
            expertise_level=updated_expertise.level,
            domain_expertise=updated_expertise.domain_expertise,
            interface_preference=updated_preferences.interface_preference,
            success_patterns=updated_patterns.success_patterns,
            failure_patterns=updated_patterns.failure_patterns,
            learning_velocity=updated_expertise.learning_velocity,
            last_updated=datetime.now()
        )
```

### 2. Expertise Analyzer

```python
class ExpertiseAnalyzer:
    """Analyzes user expertise based on configuration history and outcomes"""
    
    def analyze_expertise(self, 
                         profile: UserProfile, 
                         recent_configs: List[ConfigurationRecord]) -> ExpertiseAnalysis:
        """
        Analyze user expertise across multiple dimensions
        
        Returns:
            ExpertiseAnalysis with overall level and domain-specific expertise
        """
        # Analyze configuration complexity progression
        complexity_progression = self._analyze_complexity_progression(recent_configs)
        
        # Analyze success rates across different complexity levels
        success_rates = self._analyze_success_rates_by_complexity(recent_configs)
        
        # Analyze domain-specific expertise
        domain_expertise = self._analyze_domain_expertise(recent_configs)
        
        # Analyze learning velocity
        learning_velocity = self._analyze_learning_velocity(profile, recent_configs)
        
        # Determine overall expertise level
        overall_level = self._determine_overall_expertise_level(
            complexity_progression, success_rates, domain_expertise
        )
        
        return ExpertiseAnalysis(
            level=overall_level,
            domain_expertise=domain_expertise,
            learning_velocity=learning_velocity,
            confidence_score=self._calculate_confidence_score(recent_configs)
        )
    
    def _analyze_complexity_progression(self, configs: List[ConfigurationRecord]) -> ComplexityProgression:
        """Analyze how user's configuration complexity has evolved"""
        complexity_scores = []
        for config in sorted(configs, key=lambda x: x.created_at):
            complexity_score = self._calculate_config_complexity(config)
            complexity_scores.append((config.created_at, complexity_score))
        
        # Calculate trend
        if len(complexity_scores) < 2:
            return ComplexityProgression.STABLE
        
        recent_avg = np.mean([score for _, score in complexity_scores[-5:]])
        early_avg = np.mean([score for _, score in complexity_scores[:5]])
        
        if recent_avg > early_avg * 1.2:
            return ComplexityProgression.INCREASING
        elif recent_avg < early_avg * 0.8:
            return ComplexityProgression.DECREASING
        else:
            return ComplexityProgression.STABLE
    
    def _analyze_success_rates_by_complexity(self, configs: List[ConfigurationRecord]) -> Dict[str, float]:
        """Analyze success rates across different complexity levels"""
        complexity_buckets = {
            'simple': [],
            'moderate': [],
            'complex': []
        }
        
        for config in configs:
            complexity_score = self._calculate_config_complexity(config)
            success_score = config.outcome.success_score if config.outcome else 0.0
            
            if complexity_score < 0.3:
                complexity_buckets['simple'].append(success_score)
            elif complexity_score < 0.7:
                complexity_buckets['moderate'].append(success_score)
            else:
                complexity_buckets['complex'].append(success_score)
        
        success_rates = {}
        for bucket, scores in complexity_buckets.items():
            if scores:
                success_rates[bucket] = np.mean(scores)
            else:
                success_rates[bucket] = 0.0
        
        return success_rates
    
    def _analyze_domain_expertise(self, configs: List[ConfigurationRecord]) -> Dict[str, float]:
        """Analyze expertise in specific domains (e.g., XGBoost, PyTorch, etc.)"""
        domain_performance = {}
        
        for config in configs:
            domain = self._extract_domain_from_config(config)
            if domain not in domain_performance:
                domain_performance[domain] = []
            
            if config.outcome:
                domain_performance[domain].append(config.outcome.success_score)
        
        # Calculate expertise scores
        domain_expertise = {}
        for domain, scores in domain_performance.items():
            if len(scores) >= 3:  # Need minimum samples
                avg_success = np.mean(scores)
                consistency = 1.0 - np.std(scores)  # Higher consistency = higher expertise
                sample_size_factor = min(len(scores) / 10.0, 1.0)  # More samples = higher confidence
                
                expertise_score = (avg_success * 0.6 + consistency * 0.3 + sample_size_factor * 0.1)
                domain_expertise[domain] = min(expertise_score, 1.0)
        
        return domain_expertise
```

### 3. Preference Tracker

```python
class PreferenceTracker:
    """Tracks and analyzes user interface and configuration preferences"""
    
    def analyze_preferences(self,
                          profile: UserProfile,
                          recent_configs: List[ConfigurationRecord]) -> PreferenceAnalysis:
        """
        Analyze user preferences based on configuration history
        
        Returns:
            PreferenceAnalysis with interface and configuration preferences
        """
        # Analyze interface preferences
        interface_preference = self._analyze_interface_preferences(recent_configs)
        
        # Analyze configuration style preferences
        config_style_preferences = self._analyze_config_style_preferences(recent_configs)
        
        # Analyze optimization preferences
        optimization_preferences = self._analyze_optimization_preferences(recent_configs)
        
        return PreferenceAnalysis(
            interface_preference=interface_preference,
            config_style_preferences=config_style_preferences,
            optimization_preferences=optimization_preferences,
            confidence_score=self._calculate_preference_confidence(recent_configs)
        )
    
    def _analyze_interface_preferences(self, configs: List[ConfigurationRecord]) -> InterfacePreference:
        """Analyze which interface types user prefers"""
        interface_usage = {}
        interface_satisfaction = {}
        
        for config in configs:
            interface_type = config.metadata.interface_used
            if interface_type not in interface_usage:
                interface_usage[interface_type] = 0
                interface_satisfaction[interface_type] = []
            
            interface_usage[interface_type] += 1
            if config.outcome:
                interface_satisfaction[interface_type].append(config.outcome.user_satisfaction)
        
        # Determine preferred interface based on usage and satisfaction
        best_interface = None
        best_score = 0.0
        
        for interface_type, usage_count in interface_usage.items():
            if interface_type in interface_satisfaction and interface_satisfaction[interface_type]:
                avg_satisfaction = np.mean(interface_satisfaction[interface_type])
                usage_factor = min(usage_count / 10.0, 1.0)  # Normalize usage
                
                combined_score = avg_satisfaction * 0.7 + usage_factor * 0.3
                if combined_score > best_score:
                    best_score = combined_score
                    best_interface = interface_type
        
        return best_interface or InterfacePreference.GUIDED
```

## Data Models

```python
@dataclass
class UserProfile:
    """Comprehensive user profile"""
    user_id: str
    expertise_level: UserExpertiseLevel
    interface_preference: InterfacePreference
    domain_expertise: Dict[str, float]  # domain -> expertise score (0-1)
    success_patterns: List[SuccessPattern]
    failure_patterns: List[FailurePattern]
    learning_velocity: LearningVelocity
    created_at: datetime
    last_updated: datetime
    
    def with_updates(self, **kwargs) -> 'UserProfile':
        """Create updated profile with new values"""
        return replace(self, **kwargs)

@dataclass
class ExpertiseAnalysis:
    """Analysis of user expertise"""
    level: UserExpertiseLevel
    domain_expertise: Dict[str, float]
    learning_velocity: LearningVelocity
    confidence_score: float

@dataclass
class PreferenceAnalysis:
    """Analysis of user preferences"""
    interface_preference: InterfacePreference
    config_style_preferences: Dict[str, Any]
    optimization_preferences: Dict[str, float]
    confidence_score: float

class UserExpertiseLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class InterfacePreference(Enum):
    DECLARATIVE = "declarative"
    GUIDED = "guided"
    ADVANCED = "advanced"

class LearningVelocity(Enum):
    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"
```

## Integration with Context Analysis Framework

The User Profiler integrates with the main Context Analysis Framework through:

```python
class ContextAnalyzer:
    def __init__(self):
        self.user_profiler = UserProfiler()
        # ... other components
    
    def analyze_full_context(self, user_id: str, request: ConfigRequest) -> PipelineContext:
        # Get user profile
        user_profile = self.user_profiler.get_profile(user_id)
        
        # Use profile in context analysis
        # ...
```

## Testing Strategy

```python
class TestUserProfiler(unittest.TestCase):
    def setUp(self):
        self.user_profiler = UserProfiler()
        self.test_user_id = "test_user_123"
    
    def test_create_initial_profile(self):
        """Test initial profile creation for new user"""
        profile = self.user_profiler.get_profile(self.test_user_id)
        
        self.assertEqual(profile.user_id, self.test_user_id)
        self.assertEqual(profile.expertise_level, UserExpertiseLevel.BEGINNER)
        self.assertEqual(profile.interface_preference, InterfacePreference.GUIDED)
    
    def test_expertise_progression(self):
        """Test that expertise level increases with successful complex configurations"""
        # Create mock configuration history showing progression
        configs = self._create_mock_progression_configs()
        
        profile = UserProfile(
            user_id=self.test_user_id,
            expertise_level=UserExpertiseLevel.BEGINNER,
            # ... other fields
        )
        
        # Analyze expertise
        expertise_analysis = self.user_profiler.expertise_analyzer.analyze_expertise(
            profile, configs
        )
        
        # Should show increased expertise
        self.assertGreater(expertise_analysis.level.value, UserExpertiseLevel.BEGINNER.value)
```

## Related Documents

- **[Context Analysis Framework](./context_analysis_framework.md)** - Main framework overview
- **[Environment Detector Design](./environment_detector_design.md)** - Environment analysis component
- **[Data Source Analyzer Design](./data_source_analyzer_design.md)** - Data analysis component
- **[History Analyzer Design](./history_analyzer_design.md)** - Historical pattern analysis
