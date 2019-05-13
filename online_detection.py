import datetime
import time

import click

from lampy import helper
from lampy.detector import Detector


API_URL = 'https://ckc-emea.cisco.com/t/prague-city.com/cdp/v1/devices'


def get_pm10_measurement(sid, session, url):
    """Query API for current value on PM10 of lamp with `sid` identifier."""
    # prepare request
    body = {'Query': {'Find': {'EnvironmentData': {'sid': {'eq': sid}}}}}
    # execute request
    response = session.post(url, json=body)
    # convert it to json
    json = response.json()
    # extrat the PM10 value
    return json['Find']['Result'][0]['EnvironmentData']['airQuality']['reading']['pm10']['value']


@click.command()
@click.option('--delay', default=(60 * 15),
             help='Detection delay in seconds (defaults to 15 minutes).')
@click.option('--config', default='config.cfg', type=click.Path(exists=True),
             help='Configuration file (defaults to \'config.cfg\').')
@click.argument('sid')
def monitor_anomalies(sid, delay, config):
    """Online detection of anomalies of lamp with identifier SID."""
    # prepare session for API querying
    session = helper.setup_session(config)
    # create detector of anomalies
    detector = Detector()
    # get initial mesurement
    measurement = get_pm10_measurement(sid, session, API_URL)
    # create vector of previous measurements for prediction
    measurements = [measurement, measurement]

    # start anomaly detection
    while True:
        # get current value
        current_measurement = get_pm10_measurement(sid, session)
        # check anomaly
        is_anomaly, pred = detector.detect(measurements, current_measurement)
        if is_anomaly:
            click.echo('ANOMALY!', end=' ')
        click.echo('[' + str(datetime.datetime.now()) + '] predicted: {:.2f} true: {:.2f}'.format(pred, current_measurement))
        # update vectory of previous measurements
        measurements[0] = measurements[1]
        measurements[1] = current_measurement
        # wait for next detection
        time.sleep(delay)


if __name__ == '__main__':
    monitor_anomalies()
