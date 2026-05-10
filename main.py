
from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from mlProject.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from mlProject.pipeline.stage_04_model_trainer import ProductionModelTrainerPipeline, ModelTrainerTrainingPipeline
from mlProject.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline


def run_full_pipeline(use_production: bool = True) -> dict:
    stages = [
        ("Data Ingestion", DataIngestionTrainingPipeline),
        ("Data Validation", DataValidationTrainingPipeline),
        ("Data Transformation", DataTransformationTrainingPipeline),
        ("Model Training", ProductionModelTrainerPipeline if use_production else ModelTrainerTrainingPipeline),
        ("Model Evaluation", ModelEvaluationTrainingPipeline),
    ]

    results = {}
    for stage_name, stage_class in stages:
        try:
            logger.info(f">>>>>> stage {stage_name} started <<<<<<")
            stage_instance = stage_class()
            result = stage_instance.main()
            results[stage_name.lower().replace(" ", "_")] = {
                "status": "success",
                "result": result
            }
            logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")

        except Exception as e:
            logger.exception(f"Stage {stage_name} failed: {e}")
            results[stage_name.lower().replace(" ", "_")] = {
                "status": "failed",
                "error": str(e)
            }
            raise e

    return results


def run_training_only() -> dict:
    try:
        logger.info(">>>>> Model Training Pipeline started <<<<<<")

        logger.info(">>> Data Ingestion <<<")
        ingestion = DataIngestionTrainingPipeline()
        ingestion.main()

        logger.info(">>> Data Validation <<<")
        validation = DataValidationTrainingPipeline()
        validation.main()

        logger.info(">>> Data Transformation <<<")
        transformation = DataTransformationTrainingPipeline()
        transformation.main()

        logger.info(">>> Model Training <<<")
        trainer = ProductionModelTrainerPipeline()
        training_result = trainer.main()

        logger.info(">>> Model Evaluation <<<")
        evaluator = ModelEvaluationTrainingPipeline()
        evaluator.main()

        logger.info(">>>>> Training Pipeline completed <<<<<<\n\nx==========x")

        return {
            "status": "success",
            "training_result": training_result
        }

    except Exception as e:
        logger.exception(f"Training pipeline failed: {e}")
        raise e


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Production ML Pipeline")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'train', 'predict'],
        default='train',
    )
    parser.add_argument(
        '--production',
        action='store_true',
    )
    parser.add_argument(
        '--model-path',
        type=str,
    )

    args = parser.parse_args()

    if args.mode == 'full':
        logger.info("Running full pipeline...")
        run_full_pipeline(use_production=args.production)
    elif args.mode == 'train':
        logger.info("Running training pipeline...")
        run_training_only()
    elif args.mode == 'predict':
        logger.info("Running prediction...")
        from mlProject.pipeline.prediction import ProductionPredictionPipeline
        import numpy as np

        pipeline = ProductionPredictionPipeline(model_path=args.model_path)
        test_data = np.array([[7.4, 0.7, 0.0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]])
        prediction = pipeline.predict(test_data)
        logger.info(f"Prediction: {prediction}")