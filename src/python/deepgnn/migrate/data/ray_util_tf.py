def train_func(config: Dict):
    tf.keras.utils.set_random_seed(0)

    model = config["model"]
    tf_dataset = config["dataset"]

    model.optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

    with tf.distribute.get_strategy().scope():
        model.compile(optimizer=model.optimizer)

    for epoch in range(config["n_epochs"]):
        history = model.fit(tf_dataset, verbose=0)
        session.report(history.history)


def run_ray(**kwargs):
    trainer = TensorflowTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={**kwargs},
        run_config=RunConfig(verbose=0),
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    )
    result = trainer.fit()
