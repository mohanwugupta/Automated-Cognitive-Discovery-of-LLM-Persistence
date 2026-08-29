from .bandit import BanditRenderer
from .debugging import DebuggingRenderer
from .effort import EffortRenderer
from .foraging import ForagingRenderer
from .information_sampling import InformationSamplingRenderer
from .solvability import SolvabilityRenderer
from .waiting import WaitingRenderer


TASK_RENDERERS = {
    renderer.task_family: renderer()
    for renderer in (
        BanditRenderer,
        ForagingRenderer,
        SolvabilityRenderer,
        InformationSamplingRenderer,
        WaitingRenderer,
        EffortRenderer,
        DebuggingRenderer,
    )
}


def get_renderer(task_family: str):
    try:
        return TASK_RENDERERS[task_family]
    except KeyError as error:
        raise ValueError(f"no renderer registered for {task_family}") from error

