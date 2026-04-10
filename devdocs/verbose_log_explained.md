# Qué significa el log verbose (-v) de speechlib

Cuando corrés `python -m speechlib audio.wav -v`, el flag `-v` pone el root logger
en `DEBUG`. Esto hace que librerías internas como httpcore, httpx y speechbrain
impriman todo su tráfico. Acá te explico qué significa cada cosa.

## Fuentes del log

### httpcore.connection DEBUG

Muestra cada conexión TCP y handshake TLS que se abre hacia huggingface.co.
Líneas tipo `connect_tcp` y `start_tls`. Es el nivel más bajo de red — no sirve
para diagnosticar nada de speechlib, solo confirma que hay conectividad.

### httpcore.http11 DEBUG

Muestra el ciclo request/response HTTP/1.1 crudo: headers enviados, status code
recibido, headers de respuesta. Extremadamente verboso. Cada archivo que se
verifica contra HuggingFace Hub genera ~10 líneas de este logger.

### httpx INFO

Resumen de cada request HTTP en una línea. **Estas son las líneas útiles**:
muestran el método, URL y status code. Ejemplo:
`HTTP Request: GET https://huggingface.co/.../config.yaml 200 OK`

### speechbrain.utils.quirks INFO

Configuración interna de SpeechBrain: si JIT está habilitado, si TF32 está activo
en la GPU, etc. Sale una sola vez al inicio y se puede ignorar tranquilamente.

## Qué está pasando en esta fase

Cuando pyannote carga el pipeline con `Pipeline.from_pretrained`, verifica contra
HuggingFace Hub que los archivos del modelo estén actualizados. Los archivos que
chequea son:

- `config.yaml` — configuración del pipeline
- `segmentation/pytorch_model.bin` — modelo de segmentación de speakers
- `embedding/pytorch_model.bin` — modelo de embeddings (xvectors)
- `plda/xvec_transform.npz` — transformación pre-PLDA
- `plda/plda.npz` — modelo PLDA para scoring

Para cada archivo:

- **HTTP 200** = el archivo no cambió, se usa la versión cacheada local
- **HTTP 302** = redirect al CDN de XetHub para bajar los pesos del modelo
- **Header `X-Hub-Cache: HIT`** = HuggingFace ya tenía el archivo cacheado en su CDN

Después de la primera descarga, todo queda en cache local
(`~/.cache/huggingface/hub/`) y las verificaciones son rápidas (solo HEAD/GET con
ETag).

## Cómo reducir el ruido

El problema es que `_setup_logging` en `__main__.py` silencia urllib3, filelock,
huggingface_hub, fsspec y numba, pero **no silencia httpcore ni httpx**. Esos dos
son los que generan el 90% del spam.

La corrección es agregar `"httpcore"` y `"httpx"` a la lista de loggers silenciados
en `_setup_logging`:

```python
# En speechlib/__main__.py, línea 22
for name in ("urllib3", "filelock", "huggingface_hub", "fsspec", "numba",
             "httpcore", "httpx"):
    logging.getLogger(name).setLevel(logging.WARNING)
```

Con eso, el log verbose queda limpio: solo muestra el progreso de speechlib,
faster-whisper y pyannote, que es lo que realmente importa para diagnosticar.
