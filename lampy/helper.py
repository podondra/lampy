import click
from configparser import ConfigParser
from oauthlib.oauth2 import LegacyApplicationClient
from requests_oauthlib import OAuth2Session
import sys


def parse_config(path):
    """Parse the configuration file with ConfigParser."""
    # parse config
    cfg = ConfigParser()
    # if config file does not exist 'cfg' will be empty
    cfg.read(path)
    return cfg


def get_client_id(cfg):
    return get_from_cfg(cfg, what='client_id', err_code=1)


def get_client_secret(cfg):
    return get_from_cfg(cfg, what='client_secret', err_code=2)


def get_username_with_domain(cfg):
    return get_from_cfg(cfg, what='username_with_domain', err_code=3)


def get_password(cfg):
    return get_from_cfg(cfg, what='password', err_code=4)


def get_from_cfg(cfg, what, err_code):
    try:
        value = cfg['golemio'][what]
    except KeyError:
        click.echo('No {} has been provided'.format(what), err=True)
        sys.exit(err_code)
    return value


def get_access_token(cfg):
    # implemented according the following documentation:
    # https://requests-oauthlib.readthedocs.io/en/latest/oauth2_workflow.html#legacy-application-flow
    client_id = get_client_id(cfg)
    client_secret = get_client_secret(cfg)
    username_with_domain = get_username_with_domain(cfg)
    password = get_password(cfg)

    oauth = OAuth2Session(client=LegacyApplicationClient(client_id=client_id))
    token = oauth.fetch_token(
            token_url='https://ckc-emea.cisco.com/corev4/token',
            username=username_with_domain,
            password=password,
            client_id=client_id,
            client_secret=client_secret
            )
    return token


if __name__ == '__main__':
    cfg = parse_config('config.cfg')
    token = get_access_token(cfg)
    print(token)
