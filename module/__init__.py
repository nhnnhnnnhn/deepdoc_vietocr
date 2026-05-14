#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import io
import sys
import threading
import pdfplumber

from .ocr import OCR
from .recognizer import Recognizer
from .layout_recognizer import LayoutRecognizer4YOLOv10 as LayoutRecognizer
from .table_structure_recognizer import TableStructureRecognizer


LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
if LOCK_KEY_pdfplumber not in sys.modules:
    sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()


def init_in_out(args):
    from PIL import Image
    import os
    import traceback
    from utils.file_utils import traversal_files
    images = []
    outputs = []

    os.makedirs(args.output_dir, exist_ok=True)

    def output_folder_for(fnm):
        stem = os.path.splitext(os.path.basename(fnm))[0]
        folder = os.path.join(args.output_dir, stem)
        os.makedirs(folder, exist_ok=True)
        return folder, stem

    def pdf_pages(fnm, zoomin=3):
        nonlocal outputs, images
        with sys.modules[LOCK_KEY_pdfplumber]:
            pdf = pdfplumber.open(fnm)
            page_images = [p.to_image(resolution=72 * zoomin).annotated for i, p in
                           enumerate(pdf.pages)]

        folder, stem = output_folder_for(fnm)
        images.extend(page_images)
        for i, _ in enumerate(page_images):
            outputs.append(os.path.join(folder, f"{stem}_{i}.jpg"))
        pdf.close()

    def images_and_outputs(fnm):
        nonlocal outputs, images
        if fnm.split(".")[-1].lower() == "pdf":
            pdf_pages(fnm)
            return
        try:
            fp = open(fnm, 'rb')
            binary = fp.read()
            fp.close()
            images.append(Image.open(io.BytesIO(binary)).convert('RGB'))
            folder, _ = output_folder_for(fnm)
            outputs.append(os.path.join(folder, os.path.basename(fnm)))
        except Exception:
            traceback.print_exc()

    if os.path.isdir(args.inputs):
        for fnm in traversal_files(args.inputs):
            images_and_outputs(fnm)
    else:
        images_and_outputs(args.inputs)

    return images, outputs


__all__ = [
    "OCR",
    "Recognizer",
    "LayoutRecognizer",
    "TableStructureRecognizer",
    "init_in_out",
]
