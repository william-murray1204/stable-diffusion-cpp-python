```bash
pytest -s --maxfail=1
```

```bash
pytest tests\test_txt2img.py -s; pytest tests\test_controlnet.py -s; pytest tests\test_convert_model.py -s; pytest tests\test_flux.py -s; pytest tests\test_img2img.py -s; pytest tests\test_preprocess_canny.py -s; pytest tests\test_system_info.py -s; pytest tests\test_upscale.py -s; pytest tests\test_photomaker.py -s; pytest tests\test_inpainting.py -s; pytest tests\test_chroma.py -s; pytest tests\test_edit.py -s; pytest tests\test_vid.py -s; pytest tests/test_sd3.py -s
```
