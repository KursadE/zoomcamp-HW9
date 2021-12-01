FROM agrigorev/zoomcamp-cats-dogs-lambda:v2

RUN pip install pillow
RUN pip install \
https://github.com/alexeygrigorev/tflite-aws-lambda/blob/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl?raw=true


#RUN pip install --extra-index-url \ 
#    https://google-coral.github.io/py-repo/ tflite_runtime

COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]