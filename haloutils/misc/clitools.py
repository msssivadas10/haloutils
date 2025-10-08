import os, os.path, logging.config, click
from typing import overload, Literal

PATH_DIR    = 'DIR'
PATH_EXISTS = 'EXISTS'

def command_with_options(fn, /, **option_spec: tuple):

    import inspect, pathlib
    from typing import get_args, get_origin

    params  = inspect.signature(fn).parameters
    options = [] 
    for key in params:
        if key not in option_spec: continue
        opts, help, *flags = option_spec[key]
        p      = params[key]
        decls  = [ f"--{key}".replace('_', '-'), *opts ] 
        attrs  = {}
        optype = p.annotation
        if get_origin(optype) is list:
            optype, = get_args(optype)
            attrs.update( multiple = True, envvar = p.name.upper() )
        elif get_origin(optype) is tuple: 
            optype = get_args(optype)
        elif get_origin(optype) is Literal:
            optype = click.Choice(get_args(optype))
        elif optype is pathlib.Path:
            path_args = {}
            if PATH_DIR    in flags: path_args["file_okay"] = False
            if PATH_EXISTS in flags: path_args["exists"   ] = True
            optype = click.Path(**path_args)
        attrs.update( type = optype )
        if p.default is not p.empty: 
            attrs.update( default = p.default, required = False )
        else:
            attrs.update( required = True )
        options.append( click.option(*decls, **attrs, help = help) )
    
    for option_decorator in reversed(options): 
        fn = option_decorator(fn)

    return click.command(fn)

@overload
def build_cli(*cmds, help: str = '', version: str = '', logfn: str = '', ignore_warnings: bool = True, group: bool = False): ...
def build_cli(*cmds, **kwargs):

    assert len(cmds) > 0

    help_string     = kwargs.get("help"           , ''   ) or '' 
    version         = kwargs.get("version"        , ''   ) or '' 
    logfn           = kwargs.get("logfn"          , ''   ) or ''
    ignore_warnings = kwargs.get("ignore_warnings", True ) 
    group           = kwargs.get("group"          , False)
    
    if ignore_warnings:
        import warnings
        warnings.catch_warnings(action = "ignore") 

    # Configuring logger
    handlers= {
        "stream": {
            "level"    : "INFO", 
            "formatter": "default", 
            "class"    : "logging.StreamHandler", 
            "stream"   : "ext://sys.stdout"
        }, 
    }
    if logfn:
        fn = os.path.join( os.getcwd(), "logs", logfn + '.log' )
        if not os.path.exists( os.path.dirname(fn) ):
            os.makedirs( os.path.dirname(fn) , exist_ok = True)
        handlers["file"] = {
            "level"      : "INFO", 
            "formatter"  : "default", 
            "class"      : "logging.handlers.RotatingFileHandler", 
            "filename"   : fn, 
            "mode"       : "a", 
            "maxBytes"   : 10485760, # create a new file if size exceeds 10 MiB
            "backupCount": 4         # use maximum 4 files
        }
    logging.config.dictConfig({
        "version": 1, 
        "disable_existing_loggers": True, 
        "formatters" : { 
            "default": { "format": "[ %(asctime)s %(levelname)s %(process)d ] %(message)s" }
        }, 
        "handlers": handlers, 
        "loggers" : { 
            "root": { 
                "level"   : "INFO", 
                "handlers": list(handlers) 
            } 
        }
    })

    if len(cmds) == 1 and group == False:
        fn, option_spec = cmds[0]
        if version:
            fn = click.version_option(version, message = "%(prog)s v%(version)s")(fn)
        return command_with_options(fn, **option_spec)

    def cli(): ...
    cli.__doc__ = help_string
    if version:
        cli = click.version_option(version, message = "%(prog)s v%(version)s")(cli)
    cli = click.group(cli)
    for fn, option_spec in cmds:
        cli.add_command( command_with_options(fn, **option_spec) )

    return cli