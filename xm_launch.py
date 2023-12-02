from typing import Sequence

from absl import app
from xmanager import xm
from xmanager import xm_local

def main(argv: Sequence[str]) -> None:

  with xm_local.create_experiment(experiment_title='cifar10') as experiment:

      spec = xm.PythonContainer(
          path='/home/damian/git/spaceship_learn',
          entrypoint=xm.ModuleName('cifar10'),
      )
      [executable] = experiment.package([
      xm.Packageable(
          executable_spec=spec,
          executor_spec=xm_local.Local.Spec(),
      ),
      ])

  import itertools

  batch_sizes = [64, 1024]
  learning_rates = [0.1, 0.001]

  trials = list(
    dict([('batch_size', bs), ('learning_rate', lr)])
    for (bs, lr) in itertools.product(batch_sizes, learning_rates)
  )
  requirements = xm.JobRequirements(T)

  for hyperparameters in trials:
    experiment.add(xm.Job(
        executable=executable,
        executor=xm_local.Vertex(requirements=requirements),
        args=hyperparameters,
      ))
  
  if __name__ == '__main__':
    app.run(main)