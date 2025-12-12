FROM mambaorg/micromamba:1.5.8

WORKDIR /app

COPY environment.yml .

RUN micromamba create -y -n ai_agent_board_rules -f environment.yml \
    && micromamba clean --all --yes

ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV CONDA_DEFAULT_ENV=app
ENV PATH=/opt/conda/envs/app/bin:$PATH

COPY src ./src

CMD ["python", "-m", "boardgame_agents"]
