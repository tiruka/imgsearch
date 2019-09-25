FROM python:3.6

RUN mkdir /var/www

WORKDIR /var/www

COPY ./etc/requirements.pip ./

RUN pip install --no-cache-dir -r requirements.pip

CMD ["uwsgi","--ini","/var/www/uwsgi.ini"]