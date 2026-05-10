
from mlProject.config.configuration import ConfigurationManager
from mlProject.core.exceptions import ModelTrainingException, handle_exception
from mlProject import logger
from typing import Optional, Dict, Any


STAGE_NAME = "Model Trainer stage"


class ProductionModelTrainerPipeline:

    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.trainer = None
        self.training_result: Optional[Dict[str, Any]] = None

    def main(self) -> Dict[str, Any]:
        try:
            model_trainer_config = self.config_manager.get_model_trainer_config()

            from mlProject.components.model_trainer import ProductionModelTrainer

            self.trainer = ProductionModelTrainer(
                config=model_trainer_config,
                experiment_name="wine-quality-production",
                tracking_uri=model_trainer_config.mlflow_uri if hasattr(
                    model_trainer_config, 'mlflow_uri'
                ) else None
            )

            self.training_result = self.trainer.train(
                enable_logging=True,
                run_name=f"prod_run_{self._get_timestamp()}"
            )

            logger.info(f"模型训练完成: {self.training_result}")

            return self.training_result

        except Exception as e:
            logger.error(f"模型训练管道失败: {str(e)}")
            handle_exception(e, ModelTrainingException)
            raise

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def register_production_model(
        self,
        model_name: str = "ElasticnetProduction",
        stage: str = "Staging"
    ) -> Dict[str, Any]:
        """
        注册模型到生产环境

        Args:
            model_name: 模型名称
            stage: 模型阶段

        Returns:
            注册结果
        """
        if self.trainer is None:
            raise ModelTrainingException("模型尚未训练")

        try:
            result = self.trainer.register_model(
                model_name=model_name,
                stage=stage,
                description=f"Production model trained at {self._get_timestamp()}"
            )
            logger.info(f"模型已注册: {result}")
            return result

        except Exception as e:
            logger.error(f"模型注册失败: {str(e)}")
            raise

    def promote_to_production(self, model_name: str = "ElasticnetProduction") -> Dict[str, Any]:
        """
        将模型提升到生产环境

        Args:
            model_name: 模型名称

        Returns:
            更新结果
        """
        from mlflow.tracking import MlflowClient

        client = MlflowClient()

        try:
            # 获取最新版本
            latest_versions = client.get_latest_model_versions(model_name)
            if not latest_versions:
                raise ModelTrainingException(f"模型不存在: {model_name}")

            latest_version = latest_versions[0].version

            # 转换到生产环境
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage="Production"
            )

            result = {
                "name": model_name,
                "version": latest_version,
                "stage": "Production",
                "status": "promoted"
            }

            logger.info(f"模型已提升到生产环境: {result}")
            return result

        except Exception as e:
            logger.error(f"模型提升失败: {str(e)}")
            raise


class ModelTrainerTrainingPipeline:
    """Legacy pipeline for backward compatibility"""

    def __init__(self):
        pass

    def main(self):
        """执行模型训练"""
        pipeline = ProductionModelTrainerPipeline()
        return pipeline.main()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ProductionModelTrainerPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e