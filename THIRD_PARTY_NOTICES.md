# Credits and third-party terms

Experiment19 HMDB51 provenance: Kuehne, Jhuang, Garrote, Poggio and Serre, "HMDB: A Large Video Database for Human Motion Recognition," ICCV2011. The official Serre Lab page and pinned Serrelab distribution advertise CC BY4.0. Original unstabilized video/annotation archive identities are in `results/causal_observability_v1/source_metadata_identities.json`. Only annotations, identities and derived input-screening records are published; video archives remain local. External parent-video IDs and checkpoint-pretraining/reupload mappings are unknown. No account or click-through agreement was accepted. The verified existing X3D checkpoint/PyTorchVideo source is reused without an environment upgrade.

The [MIT LICENSE](LICENSE) covers original project code by Siyu Han. It does not relicense dependencies, datasets, source videos, or pretrained weights. Their copyright notices and terms remain separate. No dependency source code, dataset videos, or model weights are bundled in this Git repository; local downloads and installed packages retain their existing notices.

| Project or resource | Use here | Upstream information |
| --- | --- | --- |
| PyTorch | Tensor operations and CUDA inference | [Project](https://pytorch.org/), [license](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| torchvision | Pretrained ResNet18 and documented preprocessing | [ResNet18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), [license](https://github.com/pytorch/vision/blob/main/LICENSE) |
| PyAV | Python interface for sequential demuxing and decoding | [Project](https://github.com/PyAV-Org/PyAV), [license](https://github.com/PyAV-Org/PyAV/blob/main/LICENSE.txt) |
| FFmpeg | Video libraries linked by the PyAV wheel | [Project](https://ffmpeg.org/), [license information](https://ffmpeg.org/legal.html) |
| OpenMMLab / MMAction2 | Distributor of the Kinetics-400 tiny archive | [Project](https://github.com/open-mmlab/mmaction2), [license](https://github.com/open-mmlab/mmaction2/blob/main/LICENSE), [archive](https://download.openmmlab.com/mmaction/kinetics400_tiny.zip) |
| Kinetics dataset | Source of the selected validation clips | [Dataset paper](https://arxiv.org/abs/1705.06950), [dataset resources](https://github.com/cvdfoundation/kinetics-dataset) |

The dataset annotations, videos, and underlying source-video rights are not covered by this project's MIT license. Consult the dataset distributor and original video rights holders for applicable terms. Likewise, pretrained ResNet18 weights and their training-data terms remain separate from the license for this project's code. MMAction2 is credited as the archive distributor and is not installed or required to run this benchmark.

NumPy, pandas, matplotlib, Pillow, and their transitive dependencies are also used under their respective upstream licenses. Installation metadata and license files in the existing local environment have been preserved.


## Temporal recognition study

The UCF101 replication uses official [CRCV UCF101](https://www.crcv.ucf.edu/data/UCF101.php)
videos and split-1 metadata. Cite Soomro, Zamir and Shah, CRCV-TR-12-01 (2012).
The downloaded official page provides public research download links and a citation request;
no explicit video redistribution license was identified. Source-video rights remain separate
from this repository's MIT license. No account or new click-through terms are accepted and
no video bytes are redistributed. Official URLs, byte hashes and access records are retained
under results/ucf101_proxy_replication_v1/.

PyTorchVideo and X3D source are credited to Meta/Facebook Research; pinned revision f3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0 is Apache-2.0. The original LICENSE and copyright notices remain in the local upstream source cache. The official X3D_XS.pyth checkpoint is downloaded unchanged and identified by SHA256; it is not redistributed here. fvcore/iopath and their dependencies remain separately licensed. Kinetics/CVDF annotations and encoded source videos retain their original terms; no rights to third-party video content are granted by this project MIT license.
