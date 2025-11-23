#!/usr/bin/env python3
"""
Robust MLflow client with retry logic and connection management
"""
import mlflow
import mlflow.tensorflow
import time
import json
from typing import Dict, Any, Optional
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class RobustMLflowClient:
    """MLflow client with robust error handling and retry logic"""
    
    def __init__(self, tracking_uri: str, max_retries: int = 1, retry_delay: float = 2.0):
        self.tracking_uri = tracking_uri
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None
        self.current_run_id = None
        self.experiment_id = None
        self.enabled = False
        
        # Initialize connection
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize MLflow connection with retries"""
        if not self.tracking_uri or self.tracking_uri.strip() == "":
            logger.info("MLflow URI not set, running without MLflow")
            return
        
        logger.info(f"🔗 Attempting to connect to MLflow at: {self.tracking_uri}")
        mlflow.set_tracking_uri(self.tracking_uri)
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🔄 Connection attempt {attempt + 1}/{self.max_retries}...")
                self.client = mlflow.tracking.MlflowClient()
                experiments = self.client.search_experiments()
                logger.info(f"✅ MLflow connection successful, found {len(experiments)} experiments")
                self.enabled = True
                return
                
            except Exception as e:
                error_type = type(e).__name__
                logger.warning(f"MLflow connection attempt {attempt + 1}/{self.max_retries} failed:")
                logger.warning(f"   Error Type: {error_type}")
                logger.warning(f"   Error Message: {str(e)}")
                
                # Provide specific guidance based on error type
                if "Connection refused" in str(e) or "ConnectionError" in error_type:
                    logger.warning("   💡 Suggestion: Check if MLflow server is running on the specified port")
                elif "timeout" in str(e).lower():
                    logger.warning("   💡 Suggestion: MLflow server may be slow to respond, consider increasing timeout")
                elif "404" in str(e) or "Not Found" in str(e):
                    logger.warning("   💡 Suggestion: Check if MLflow server URL is correct")
                
                if attempt < self.max_retries - 1:
                    logger.info(f"   ⏳ Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.info("   ⚡ Fast failure mode: No retries, switching to fallback metrics")
        
        logger.error("❌ Failed to connect to MLflow (fast failure mode)")
        logger.error("   📋 Quick troubleshooting:")
        logger.error("   1. Check port-forward: kubectl port-forward -n mlops-fl svc/mlflow 5000:5000")
        logger.error("   2. Test connectivity: curl http://localhost:5000/health")
        logger.error("   3. Verify environment: echo $MLFLOW_TRACKING_URI")
        logger.error("   ⚡ Fast failure enabled - training continues with local fallback metrics")
        self.enabled = False
    
    def setup_experiment(self, experiment_name: str) -> Optional[str]:
        """Setup or create MLflow experiment with proper error handling"""
        if not self.enabled:
            return None
        
        for attempt in range(self.max_retries):
            try:
                # Try to get existing experiment first
                try:
                    experiment = self.client.get_experiment_by_name(experiment_name)
                    if experiment and experiment.lifecycle_stage != "deleted":
                        self.experiment_id = experiment.experiment_id
                        mlflow.set_experiment(experiment_name)
                        logger.info(f"✅ Using existing experiment: {experiment_name} (ID: {self.experiment_id})")
                        return self.experiment_id
                except Exception:
                    # Experiment doesn't exist, we'll create it
                    pass
                
                # Create new experiment
                try:
                    self.experiment_id = self.client.create_experiment(experiment_name)
                    mlflow.set_experiment(experiment_name)
                    logger.info(f"✅ Created new experiment: {experiment_name} (ID: {self.experiment_id})")
                    return self.experiment_id
                except Exception as create_error:
                    if "already exists" in str(create_error).lower():
                        # Race condition - experiment was created between our check and create
                        logger.info(f"⚠️ Experiment {experiment_name} was created by another process, retrying...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        raise create_error
                        
            except Exception as e:
                logger.warning(f"Experiment setup attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.info("   ⚡ Fast failure: No more retries for experiment setup")
        
        logger.error(f"❌ Failed to setup experiment (fast failure mode)")
        self.enabled = False
        return None
    
    @contextmanager
    def start_run(self, run_name: str):
        """Context manager for MLflow runs with robust error handling"""
        run = None
        run_started = False
        
        if not self.enabled:
            logger.info("MLflow disabled, running without logging")
            yield None
            return
        
        try:
            # Start MLflow run with retries and validation
            for attempt in range(self.max_retries):
                try:
                    run = mlflow.start_run(run_name=run_name)
                    self.current_run_id = run.info.run_id
                    
                    # Validate run was actually created and is accessible
                    try:
                        run_info = self.client.get_run(self.current_run_id)
                        if run_info and run_info.info.status == "RUNNING":
                            run_started = True
                            logger.info(f"✅ MLflow run started and validated: {self.current_run_id}")
                            break
                        else:
                            logger.warning(f"Run created but not accessible: {self.current_run_id}")
                            raise Exception("Run validation failed")
                    except Exception as validation_error:
                        logger.warning(f"Run validation failed: {validation_error}")
                        # Try to end the problematic run
                        try:
                            mlflow.end_run()
                        except:
                            pass
                        raise validation_error
                    
                except Exception as e:
                    logger.warning(f"Failed to start run (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
            
            if not run_started:
                logger.error("❌ Failed to start MLflow run (fast failure mode)")
                self.enabled = False
                yield None
                return
            
            yield run
            
        except Exception as e:
            logger.error(f"❌ Error during MLflow run: {e}")
            self.enabled = False
            
        finally:
            # Clean up run with validation
            if run_started and self.current_run_id:
                try:
                    # Verify run still exists before ending
                    try:
                        self.client.get_run(self.current_run_id)
                        mlflow.end_run()
                        logger.info("✅ MLflow run ended successfully")
                    except Exception as get_error:
                        if "not found" in str(get_error).lower():
                            logger.warning(f"⚠️ Run {self.current_run_id} was already ended or lost")
                        else:
                            logger.warning(f"⚠️ Failed to validate run before ending: {get_error}")
                            # Try to end anyway
                            mlflow.end_run()
                except Exception as e:
                    logger.warning(f"⚠️ Failed to end MLflow run cleanly: {e}")
                finally:
                    self.current_run_id = None
    
    def log_params(self, params: Dict[str, Any]) -> bool:
        """Log parameters with retry logic and run validation"""
        if not self.enabled or not self.current_run_id:
            return False
        
        for attempt in range(self.max_retries):
            try:
                # Validate run exists before logging
                if not self._validate_run_exists():
                    logger.error(f"❌ Run {self.current_run_id} no longer exists, cannot log params")
                    self.enabled = False
                    return False
                
                mlflow.log_params(params)
                logger.info(f"✅ Logged {len(params)} parameters")
                return True
                
            except Exception as e:
                logger.warning(f"Failed to log params (attempt {attempt + 1}): {e}")
                if "not found" in str(e).lower():
                    logger.error(f"❌ Run {self.current_run_id} was lost during param logging")
                    self.enabled = False
                    return False
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        logger.error("❌ Failed to log parameters (fast failure mode)")
        return False
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> bool:
        """Log metrics with retry logic and run validation"""
        if not self.enabled or not self.current_run_id:
            return False
        
        for attempt in range(self.max_retries):
            try:
                # Validate run exists before logging
                if not self._validate_run_exists():
                    logger.error(f"❌ Run {self.current_run_id} no longer exists, cannot log metrics")
                    self.enabled = False
                    return False
                
                mlflow.log_metrics(metrics, step=step)
                logger.info(f"✅ Logged {len(metrics)} metrics" + (f" at step {step}" if step else ""))
                return True
                
            except Exception as e:
                logger.warning(f"Failed to log metrics (attempt {attempt + 1}): {e}")
                if "not found" in str(e).lower():
                    logger.error(f"❌ Run {self.current_run_id} was lost during metrics logging")
                    self.enabled = False
                    return False
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        logger.error("❌ Failed to log metrics (fast failure mode)")
        return False
    
    def _validate_run_exists(self) -> bool:
        """Validate that the current run still exists"""
        if not self.current_run_id:
            return False
        
        try:
            run_info = self.client.get_run(self.current_run_id)
            return run_info and run_info.info.status == "RUNNING"
        except Exception:
            return False
    
    def log_model(self, model, artifact_path: str = "model", 
                  registered_model_name: Optional[str] = None) -> bool:
        """Log model with retry logic and run validation"""
        if not self.enabled or not self.current_run_id:
            return False
        
        for attempt in range(self.max_retries):
            try:
                # Validate run exists before logging
                if not self._validate_run_exists():
                    logger.error(f"❌ Run {self.current_run_id} no longer exists, cannot log model")
                    self.enabled = False
                    return False
                
                mlflow.tensorflow.log_model(
                    model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name
                )
                logger.info(f"✅ Logged model to {artifact_path}")
                return True
                
            except Exception as e:
                logger.warning(f"Failed to log model (attempt {attempt + 1}): {e}")
                if "not found" in str(e).lower():
                    logger.error(f"❌ Run {self.current_run_id} was lost during model logging")
                    self.enabled = False
                    return False
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        logger.error("❌ Failed to log model (fast failure mode)")
        return False
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> bool:
        """Log artifact with retry logic"""
        if not self.enabled or not self.current_run_id:
            return False
        
        for attempt in range(self.max_retries):
            try:
                mlflow.log_artifact(local_path, artifact_path=artifact_path)
                logger.info(f"✅ Logged artifact: {local_path}")
                return True
                
            except Exception as e:
                logger.warning(f"Failed to log artifact (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        logger.error("❌ Failed to log artifact (fast failure mode)")
        return False
    
    def set_tags(self, tags: Dict[str, str]) -> bool:
        """Set tags with retry logic"""
        if not self.enabled or not self.current_run_id:
            return False
        
        for attempt in range(self.max_retries):
            try:
                mlflow.set_tags(tags)
                logger.info(f"✅ Set {len(tags)} tags")
                return True
                
            except Exception as e:
                logger.warning(f"Failed to set tags (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        logger.error("❌ Failed to set tags (fast failure mode)")
        return False
    
    def is_enabled(self) -> bool:
        """Check if MLflow is enabled and working"""
        return self.enabled
    
    def get_run_id(self) -> Optional[str]:
        """Get current run ID"""
        return self.current_run_id