build_container_local:
	docker build --tag=${IMAGE}:dev .

run_container_local:
	docker run -it -e PORT=8000 -p 8080:8000 ${IMAGE}:dev

build_for_production:
	docker build \
		--platform linux/amd64 \
    -t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod \
		.

push_image_production:
	docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod

deploy_to_cloud_run:
	gcloud run deploy climate-resilience-app \
		--project=$(GCP_PROJECT) \
		--image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod \
		--memory ${MEMORY} \
		--region ${GCP_REGION} \
		--set-secrets GEMINI_API_KEY=gemini-api-key:latest

run_preprocess_pipeline:
	python utils/preprocessing/pipeline.py

run_prediction_model:

	python model/prediction/ES_all_indicators.py

run_save_model_output:

	python model/prediction/save_outputs.py

run_feature_importance:
	python model/feature_importance/random_forest_pipeline.py
