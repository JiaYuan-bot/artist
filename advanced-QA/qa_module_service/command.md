
1.
gcloud auth application-default login

2.
LD_PRELOAD="/root/.pyenv/versions/3.12.5/lib/python3.12/site-packages/chroma_hnswlib.libs/libgomp-d22c30c5.so.1.0.0" python main.py

3.

WARNING:langchain_core.language_models.llms:Retrying langchain_google_vertexai.llms._completion_with_retry.<locals>._completion_with_retry_inner in 4.0 seconds as it raised ResourceExhausted: 429 received metadata size exceeds soft limit (28059 vs. 16384);  :path:103B :authority:79B :method:43B :scheme:44B content-type:60B te:42B grpc-accept-encoding:75B user-agent:100B grpc-trace-bin:103B pc-low-fwd-bin:77B x-goog-request-params:148B x-goog-api-client:23688B x-goog-api-client:60B authorization:306B x-goog-user-project:58B x-google-gfe-frontline-info:911B x-google-gfe-frontline-proto:119B x-google-gfe-timestamp-trace:76B x-google-gfe-verified-user-ip:74B x-gfe-signed-request-headers:508B x-google-gfe-location-info:80B x-gfe-ssl:44B x-google-gfe-tls-base64urlclienthelloprotobuf:299B x-user-ip:54B x-google-service:115B x-google-gfe-service-trace:125B x-google-dos-service-trace:156B x-google-gfe-backend-timeout-ms:71B accept-encoding:56B x-google-peer-delegation-chain-bin:92B x-google-request-uid:143B x-google-dappertraceinfo:111B.


4. python -m venv .../pytorch_env
5. source .../pytorch_env/bin/activate

4.  conda create --name chess-env python=3.12.5
    conda activate chess-env


