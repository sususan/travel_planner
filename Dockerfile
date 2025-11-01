# Dockerfile
FROM public.ecr.aws/lambda/python:3.12

# Set writable data dirs used by crewai (Lambda /var/task is read-only; /tmp is writable)
ENV CREWAI_DATA_DIR=/tmp/crewai_data
ENV XDG_DATA_HOME=/tmp

WORKDIR /var/app

# Copy application code
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
COPY planner_agent /var/app/planner_agent
COPY planner_agent/template.yaml /var/app/planner_agent/
#COPY planner_agent/ ${LAMBDA_TASK_ROOT}/

ENV PYTHONPATH /var/app

# Create the data directory in the image (it will also exist at runtime in /tmp)
RUN mkdir -p /tmp/crewai_data \
    && chmod 777 /tmp/crewai_data

# Install dependencies into the Lambda task root so they are available at runtime
RUN python -m pip install --upgrade pip \
    && if [ -s "${LAMBDA_TASK_ROOT}/requirements.txt" ]; then pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt -t ${LAMBDA_TASK_ROOT}; fi \
    && rm -f ${LAMBDA_TASK_ROOT}/requirements.txt

# Command: module.function — Lambda runtime looks for this
# keep as you had it if that's your handler path
CMD ["planner_agent.lambda.planner_handler.lambda_handler"]
