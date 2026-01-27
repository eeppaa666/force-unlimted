from ik.src.ik_processor_base import IKProcessor
from ik.src.unitree.g1_29_ik_processor import G129IkProcessor
from ik.src.fourier.gr1t1_processor import Gr1T1Processor
from typing import Dict

IK_PROCESSOR_MAP: Dict[str, IKProcessor] = {
    "unitree_g1_29": G129IkProcessor,
    "fourier_gr1t1": Gr1T1Processor,
}
