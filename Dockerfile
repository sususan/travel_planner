# Dockerfile
# Use AWS Lambda Python 3.13 base image (public ECR)
FROM public.ecr.aws/lambda/python:3.13

# Copy application code
COPY planner_agent/ ${LAMBDA_TASK_ROOT}/planner_agent/
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install dependencies into the Lambda task root so they are available at runtime
RUN python -m pip install --upgrade pip \
    && if [ -s "${LAMBDA_TASK_ROOT}/requirements.txt" ]; then pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt -t ${LAMBDA_TASK_ROOT}; fi \
    && rm -f ${LAMBDA_TASK_ROOT}/requirements.txt

# (Optional) Expose port for local testing with RIC (not required in AWS)
# EXPOSE 8080

# Command: module.function — Lambda runtime looks for this
CMD ["planner_agent.lambda.handler.s3_event_handler"]
