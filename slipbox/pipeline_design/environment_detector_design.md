# Environment Detector Design

## Overview

The Environment Detector is a core component of the [Context Analysis Framework](./context_analysis_framework.md) that detects and analyzes deployment environment characteristics. It provides information about resource constraints, security requirements, and deployment patterns to inform configuration decisions.

## Purpose and Responsibilities

1. **AWS Environment Analysis**: Detect AWS-specific environment details and capabilities
2. **Resource Constraint Detection**: Analyze available resources and quotas
3. **Security Requirement Analysis**: Identify security constraints and requirements
4. **Cost Constraint Assessment**: Understand budget limitations and cost optimization needs
5. **Deployment Pattern Recognition**: Identify common deployment patterns and preferences

## Core Components

### 1. Environment Detector (Main Component)

```python
class EnvironmentDetector:
    """
    Detects and analyzes deployment environment characteristics
    
    Provides information about resource constraints, security requirements,
    and deployment patterns to inform configuration decisions
    """
    
    def __init__(self):
        self.aws_analyzer = AWSEnvironmentAnalyzer()
        self.resource_analyzer = ResourceAnalyzer()
        self.security_analyzer = SecurityAnalyzer()
        self.cost_analyzer = CostAnalyzer()
    
    def detect_environment(self, request: ConfigRequest) -> EnvironmentCharacteristics:
        """
        Detect comprehensive environment characteristics
        
        Args:
            request: Configuration request with environment hints
            
        Returns:
            EnvironmentCharacteristics with resource, security, and cost info
        """
        # Detect AWS environment details
        aws_environment = self.aws_analyzer.analyze_aws_environment(request)
        
        # Analyze resource constraints
        resource_constraints = self.resource_analyzer.analyze_constraints(
            aws_environment, request
        )
        
        # Analyze security requirements
        security_requirements = self.security_analyzer.analyze_requirements(
            aws_environment, request
        )
        
        # Analyze cost constraints
        cost_constraints = self.cost_analyzer.analyze_constraints(
            aws_environment, request
        )
        
        return EnvironmentCharacteristics(
            aws_environment=aws_environment,
            resource_constraints=resource_constraints,
            security_requirements=security_requirements,
            cost_constraints=cost_constraints,
            deployment_pattern=self._detect_deployment_pattern(request),
            performance_requirements=self._detect_performance_requirements(request)
        )
    
    def _detect_deployment_pattern(self, request: ConfigRequest) -> DeploymentPattern:
        """Detect deployment pattern from request characteristics"""
        # Analyze request characteristics to determine pattern
        if request.batch_size and request.batch_size > 1000:
            return DeploymentPattern.BATCH_PROCESSING
        elif request.real_time_inference:
            return DeploymentPattern.REAL_TIME_INFERENCE
        elif request.model_training:
            return DeploymentPattern.MODEL_TRAINING
        else:
            return DeploymentPattern.GENERAL_PURPOSE
    
    def _detect_performance_requirements(self, request: ConfigRequest) -> PerformanceRequirements:
        """Detect performance requirements from request"""
        return PerformanceRequirements(
            latency_requirement=request.max_latency_ms,
            throughput_requirement=request.min_throughput,
            availability_requirement=request.availability_target or 0.99,
            scalability_requirement=request.auto_scaling_enabled or False
        )
```

### 2. AWS Environment Analyzer

```python
class AWSEnvironmentAnalyzer:
    """Analyzes AWS-specific environment characteristics"""
    
    def __init__(self):
        self.boto3_session = boto3.Session()
        self.sts_client = self.boto3_session.client('sts')
        self.ec2_client = None
        self.iam_client = None
        self.service_quotas_client = None
    
    def analyze_aws_environment(self, request: ConfigRequest) -> AWSEnvironment:
        """
        Analyze AWS environment details
        
        Returns:
            AWSEnvironment with region, account, and service availability info
        """
        # Detect region from request or current session
        region = self._detect_region(request)
        
        # Initialize clients for detected region
        self._initialize_clients(region)
        
        # Detect account information
        account_info = self._detect_account_info()
        
        # Analyze service availability and quotas
        service_availability = self._analyze_service_availability(region)
        
        # Analyze resource quotas
        resource_quotas = self._analyze_resource_quotas(region, account_info)
        
        return AWSEnvironment(
            region=region,
            account_id=account_info.account_id,
            account_type=account_info.account_type,
            service_availability=service_availability,
            resource_quotas=resource_quotas,
            vpc_configuration=self._analyze_vpc_configuration(region),
            iam_capabilities=self._analyze_iam_capabilities()
        )
    
    def _detect_region(self, request: ConfigRequest) -> str:
        """Detect AWS region from request or session"""
        if request.aws_region:
            return request.aws_region
        
        # Try to get from session
        session_region = self.boto3_session.region_name
        if session_region:
            return session_region
        
        # Default to us-east-1
        return 'us-east-1'
    
    def _detect_account_info(self) -> AccountInfo:
        """Detect AWS account information"""
        try:
            identity = self.sts_client.get_caller_identity()
            account_id = identity['Account']
            
            # Determine account type based on account ID patterns or other heuristics
            account_type = self._determine_account_type(account_id)
            
            return AccountInfo(
                account_id=account_id,
                account_type=account_type,
                user_arn=identity.get('Arn'),
                user_id=identity.get('UserId')
            )
        except Exception as e:
            # Handle cases where AWS credentials are not available
            return AccountInfo(
                account_id="unknown",
                account_type=AccountType.UNKNOWN,
                user_arn=None,
                user_id=None
            )
    
    def _analyze_service_availability(self, region: str) -> ServiceAvailability:
        """Analyze which AWS services are available in the region"""
        available_services = {}
        
        # Check SageMaker availability
        try:
            sagemaker_client = self.boto3_session.client('sagemaker', region_name=region)
            sagemaker_client.list_domains(MaxResults=1)
            available_services['sagemaker'] = True
        except Exception:
            available_services['sagemaker'] = False
        
        # Check other relevant services
        services_to_check = ['s3', 'ec2', 'lambda', 'batch', 'ecs', 'eks']
        for service in services_to_check:
            available_services[service] = self._check_service_availability(service, region)
        
        return ServiceAvailability(
            available_services=available_services,
            region_capabilities=self._get_region_capabilities(region)
        )
    
    def _analyze_resource_quotas(self, region: str, account_info: AccountInfo) -> ResourceQuotas:
        """Analyze resource quotas and limits"""
        quotas = {}
        
        try:
            if not self.service_quotas_client:
                self.service_quotas_client = self.boto3_session.client(
                    'service-quotas', region_name=region
                )
            
            # Get SageMaker quotas
            sagemaker_quotas = self._get_sagemaker_quotas()
            quotas.update(sagemaker_quotas)
            
            # Get EC2 quotas
            ec2_quotas = self._get_ec2_quotas()
            quotas.update(ec2_quotas)
            
        except Exception as e:
            # Fallback to default quotas if service quotas API is not available
            quotas = self._get_default_quotas(account_info.account_type)
        
        return ResourceQuotas(
            quotas=quotas,
            current_usage=self._get_current_resource_usage(region)
        )
```

### 3. Resource Analyzer

```python
class ResourceAnalyzer:
    """Analyzes resource constraints and availability"""
    
    def analyze_constraints(self,
                          aws_environment: AWSEnvironment,
                          request: ConfigRequest) -> ResourceConstraints:
        """
        Analyze resource constraints based on environment and request
        
        Returns:
            ResourceConstraints with compute, storage, and network limits
        """
        # Analyze compute constraints
        compute_constraints = self._analyze_compute_constraints(
            aws_environment, request
        )
        
        # Analyze storage constraints
        storage_constraints = self._analyze_storage_constraints(
            aws_environment, request
        )
        
        # Analyze network constraints
        network_constraints = self._analyze_network_constraints(
            aws_environment, request
        )
        
        return ResourceConstraints(
            compute=compute_constraints,
            storage=storage_constraints,
            network=network_constraints,
            memory_limits=self._analyze_memory_limits(aws_environment),
            concurrent_jobs_limit=self._analyze_concurrent_jobs_limit(aws_environment)
        )
    
    def _analyze_compute_constraints(self,
                                   aws_environment: AWSEnvironment,
                                   request: ConfigRequest) -> ComputeConstraints:
        """Analyze compute resource constraints"""
        # Get available instance types
        available_instances = self._get_available_instance_types(
            aws_environment.region, aws_environment.service_availability
        )
        
        # Analyze quota limits
        quota_limits = self._extract_compute_quotas(aws_environment.resource_quotas)
        
        # Determine optimal instance recommendations
        recommended_instances = self._recommend_instances_for_workload(
            request, available_instances, quota_limits
        )
        
        return ComputeConstraints(
            available_instance_types=available_instances,
            quota_limits=quota_limits,
            recommended_instances=recommended_instances,
            spot_availability=self._check_spot_availability(aws_environment.region)
        )
    
    def _get_available_instance_types(self,
                                    region: str,
                                    service_availability: ServiceAvailability) -> List[str]:
        """Get list of available instance types for the region"""
        if not service_availability.available_services.get('sagemaker', False):
            return []
        
        # Common SageMaker instance types by region
        standard_instances = [
            'ml.t3.medium', 'ml.t3.large', 'ml.t3.xlarge',
            'ml.m5.large', 'ml.m5.xlarge', 'ml.m5.2xlarge', 'ml.m5.4xlarge',
            'ml.c5.large', 'ml.c5.xlarge', 'ml.c5.2xlarge', 'ml.c5.4xlarge',
            'ml.r5.large', 'ml.r5.xlarge', 'ml.r5.2xlarge', 'ml.r5.4xlarge'
        ]
        
        # GPU instances (may not be available in all regions)
        gpu_instances = [
            'ml.p3.2xlarge', 'ml.p3.8xlarge', 'ml.p3.16xlarge',
            'ml.g4dn.xlarge', 'ml.g4dn.2xlarge', 'ml.g4dn.4xlarge'
        ]
        
        # Check which instances are actually available
        available_instances = []
        for instance in standard_instances + gpu_instances:
            if self._is_instance_available_in_region(instance, region):
                available_instances.append(instance)
        
        return available_instances
```

### 4. Security Analyzer

```python
class SecurityAnalyzer:
    """Analyzes security requirements and constraints"""
    
    def analyze_requirements(self,
                           aws_environment: AWSEnvironment,
                           request: ConfigRequest) -> SecurityRequirements:
        """
        Analyze security requirements based on environment and request
        
        Returns:
            SecurityRequirements with encryption, network, and access controls
        """
        # Analyze encryption requirements
        encryption_requirements = self._analyze_encryption_requirements(
            aws_environment, request
        )
        
        # Analyze network security requirements
        network_security = self._analyze_network_security_requirements(
            aws_environment, request
        )
        
        # Analyze access control requirements
        access_control = self._analyze_access_control_requirements(
            aws_environment, request
        )
        
        return SecurityRequirements(
            encryption=encryption_requirements,
            network_security=network_security,
            access_control=access_control,
            compliance_requirements=self._detect_compliance_requirements(request),
            data_classification=self._classify_data_sensitivity(request)
        )
    
    def _analyze_encryption_requirements(self,
                                       aws_environment: AWSEnvironment,
                                       request: ConfigRequest) -> EncryptionRequirements:
        """Analyze encryption requirements"""
        # Default to encryption enabled for security
        at_rest_required = True
        in_transit_required = True
        
        # Check if request specifies encryption requirements
        if hasattr(request, 'encryption_required'):
            at_rest_required = request.encryption_required
            in_transit_required = request.encryption_required
        
        # Check for compliance requirements that mandate encryption
        if self._has_compliance_requirements(request):
            at_rest_required = True
            in_transit_required = True
        
        return EncryptionRequirements(
            at_rest_required=at_rest_required,
            in_transit_required=in_transit_required,
            kms_key_required=self._requires_customer_managed_keys(request),
            encryption_algorithm_requirements=self._get_encryption_algorithm_requirements(request)
        )
```

## Data Models

```python
@dataclass
class EnvironmentCharacteristics:
    """Environment characteristics and constraints"""
    aws_environment: AWSEnvironment
    resource_constraints: ResourceConstraints
    security_requirements: SecurityRequirements
    cost_constraints: CostConstraints
    deployment_pattern: DeploymentPattern
    performance_requirements: PerformanceRequirements

@dataclass
class AWSEnvironment:
    """AWS-specific environment information"""
    region: str
    account_id: str
    account_type: AccountType
    service_availability: ServiceAvailability
    resource_quotas: ResourceQuotas
    vpc_configuration: Optional[VPCConfiguration]
    iam_capabilities: IAMCapabilities

@dataclass
class ResourceConstraints:
    """Resource constraints and limits"""
    compute: ComputeConstraints
    storage: StorageConstraints
    network: NetworkConstraints
    memory_limits: Dict[str, int]
    concurrent_jobs_limit: int

@dataclass
class SecurityRequirements:
    """Security requirements and constraints"""
    encryption: EncryptionRequirements
    network_security: NetworkSecurityRequirements
    access_control: AccessControlRequirements
    compliance_requirements: List[str]
    data_classification: DataClassification

class AccountType(Enum):
    PERSONAL = "personal"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"
    UNKNOWN = "unknown"

class DeploymentPattern(Enum):
    BATCH_PROCESSING = "batch_processing"
    REAL_TIME_INFERENCE = "real_time_inference"
    MODEL_TRAINING = "model_training"
    GENERAL_PURPOSE = "general_purpose"
```

## Integration with Context Analysis Framework

The Environment Detector integrates with the main Context Analysis Framework through:

```python
class ContextAnalyzer:
    def __init__(self):
        self.environment_detector = EnvironmentDetector()
        # ... other components
    
    def analyze_full_context(self, user_id: str, request: ConfigRequest) -> PipelineContext:
        # Detect environment characteristics
        environment = self.environment_detector.detect_environment(request)
        
        # Use environment in context analysis
        # ...
```

## Testing Strategy

```python
class TestEnvironmentDetector(unittest.TestCase):
    def setUp(self):
        self.environment_detector = EnvironmentDetector()
        self.mock_request = ConfigRequest(
            pipeline_type="xgboost_training",
            aws_region="us-west-2"
        )
    
    def test_detect_aws_environment(self):
        """Test AWS environment detection"""
        environment = self.environment_detector.detect_environment(self.mock_request)
        
        self.assertIsNotNone(environment.aws_environment)
        self.assertEqual(environment.aws_environment.region, "us-west-2")
    
    def test_resource_constraint_analysis(self):
        """Test resource constraint analysis"""
        environment = self.environment_detector.detect
