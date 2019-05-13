from configparser import ConfigParser
from functools import partial
import sys

from oauthlib.oauth2 import LegacyApplicationClient
from requests import Session
from requests_oauthlib import OAuth2Session


TOKEN_URL = 'https://ckc-emea.cisco.com/corev4/token'


def token_updater(token):
    return


def setup_session(config_file):
    """Setup session so that it can be used for data retrieval."""
    # parse config
    cfg = parse_config(config_file)
    # get access token
    token = get_token(cfg)
    # setup authorization
    client_id = get_client_id(cfg)
    client_secret = get_client_secret(cfg)
    extra = {
        'client_secret': client_secret,
        'client_id': client_id
    }
    session = OAuth2Session(client_id,
                            token=token,
                            auto_refresh_kwargs=extra,
                            auto_refresh_url=TOKEN_URL,
                            token_updater=token_updater)
    return session


def parse_config(path):
    """Parse the configuration file with ConfigParser."""
    # parse config
    cfg = ConfigParser()
    # if config file does not exist 'cfg' will be empty
    cfg.read(path)
    return cfg


def get_client_id(cfg):
    """Get client ID from config."""
    return get_from_cfg(cfg, what='client_id', err_code=1)


def get_client_secret(cfg):
    """Get client secret from config."""
    return get_from_cfg(cfg, what='client_secret', err_code=2)


def get_username_with_domain(cfg):
    """Get username with domain from config."""
    return get_from_cfg(cfg, what='username_with_domain', err_code=3)


def get_password(cfg):
    """Get password with domain from config."""
    return get_from_cfg(cfg, what='password', err_code=4)


def get_from_cfg(cfg, what, err_code):
    """Get `what` from config or exit on error."""
    try:
        value = cfg['golemio'][what]
    except KeyError:
        print('No {} has been provided'.format(what), file=sys.stderr)
        sys.exit(err_code)
    return value


def get_token(cfg):
    """Get access token for the Golemio API."""
    # implemented according the following documentation:
    # https://requests-oauthlib.readthedocs.io/en/latest/oauth2_workflow.html#legacy-application-flow
    client_id = get_client_id(cfg)
    client_secret = get_client_secret(cfg)
    username_with_domain = get_username_with_domain(cfg)
    password = get_password(cfg)

    oauth = OAuth2Session(client=LegacyApplicationClient(client_id=client_id))
    token = oauth.fetch_token(
            token_url=TOKEN_URL,
            username=username_with_domain,
            password=password,
            client_id=client_id,
            client_secret=client_secret
            )
    return token


def token_auth(req, token):
    """Token authorization handler."""
    req.headers['Authorization'] = 'Bearer ' + token
    return req
