#!/bin/env python3

import click
import haloutils.galaxy_catalog_generator
import haloutils.correlation2
import haloutils.mass_function

def with_logging(fn):

    import os, os.path, logging.config, functools

    def get_boolean_envvar(var, default=None):
        return {
            '0': False, 'false': False, 'no' : False, 'n': False,
            '1': True , 'true' : True , 'yes': True , 'y': True ,
        }.get( os.environ.get(var, '').strip().lower(), default )

    def configure_logging(name):

        if get_boolean_envvar("DISABLE_ALL_LOGS", False): return

        filename = os.path.join( os.getcwd(), "logs",  f'{name}.log' )
        os.makedirs( os.path.dirname(filename) , exist_ok = True)
        handlers = {
            "file": {
                "level"      : "INFO", 
                "formatter"  : "default", 
                "class"      : "logging.handlers.RotatingFileHandler", 
                "filename"   : filename, 
                "mode"       : "a", 
                "maxBytes"   : 10485760, # create a new file if size exceeds 10 MiB
                "backupCount": 4         # use maximum 4 files
            }, 
            "stream": {
                "level"    : "INFO", 
                "formatter": "default", 
                "class"    : "logging.StreamHandler", 
                "stream"   : "ext://sys.stdout"
            }
        }
        if get_boolean_envvar("DISABLE_STREAM_LOGS", False): handlers.pop("stream") 
        
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
        return

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):

        if get_boolean_envvar("IGNORE_WARNINGS", True):
            import warnings
            warnings.catch_warnings( action="ignore" )

        configure_logging( fn.__name__ )
        return fn( *args, **kwargs )
    
    return wrapper


@click.group
def cli(): pass
def add_command(fn, options, help="", version=""):

    import inspect, typing, pathlib
    
    def _optname(n): return f"--{n}".replace('_', '-')

    option_decorators = []
    for name, p in inspect.signature( fn ).parameters.items():
        if name not in options: continue
        kwargs, ( args, help_text, type_flag ) = {}, options[ name ]
        option_type = p.annotation 
        assert option_type is not p.empty 
        if typing.get_origin( option_type ) is list:
            kwargs["multiple"] = True
            kwargs["envvar"  ] = name.upper()
            option_type,       = typing.get_args( option_type )
        elif typing.get_origin( option_type ) is tuple: 
            option_type = typing.get_args( option_type )
        elif typing.get_origin( option_type ) is typing.Literal:
            option_type = click.Choice( typing.get_args( option_type ) )
        elif option_type is pathlib.Path:
            if   type_flag == "dir?": option_type = click.Path( file_okay=False ) 
            elif type_flag == "dir" : option_type = click.Path( file_okay=False, exists=True ) 
            else:                     option_type = click.Path()
        kwargs["type"] = option_type
        if p.default is not p.empty:
            kwargs["default" ] = p.default
        else:
            kwargs["required"] = True
        kwargs["help"] = help_text
        option_decorators.append( click.option( _optname(name), *args, **kwargs ) )        

    fn = with_logging(fn)
    if version: fn = click.version_option( version, message="%(prog)s v%(version)s" )( fn )
    if help   : fn.__doc__ = help
    for option_decorator in reversed(option_decorators): fn = option_decorator( fn )

    return cli.add_command( click.command(fn) )
 
add_command(
    haloutils.galaxy_catalog_generator.galaxy_catalog_generator,
    help    = "Generate galaxy catalogs based on a halo catalog and halo model",
    version = "",
    options = {
        "simname"     : ( ["-s" ], "Name of simulation"             , ""     ),
        "redshift"    : ( ["-z" ], "Redshift value"                 , ""     ),
        "mmin"        : ( ["-mm"], "Central galaxy threshold mass"  , ""     ),
        "m0"          : ( ["-m0"], "Satellite galaxy threshold"     , ""     ),
        "m1"          : ( ["-m1"], "Satellite count amplitude"      , ""     ),
        "sigma_m"     : ( ["-w" ], "Central galaxy width parameter" , ""     ),
        "alpha"       : ( ["-a" ], "Satellite power law count index", ""     ),
        "scale_shmf"  : ( ["-b" ], "SHMF scale parameter"           , ""     ),
        "slope_shmf"  : ( ["-j" ], "SHMF slope parameter"           , ""     ),
        "filter_fn"   : ( ["-f" ], "Filter function for variance"   , ""     ),
        "sigma_size"  : ( ["-k" ], "Size of variance table"         , ""     ),
        "nthreads"    : ( ["-n" ], "Number of threads to use"       , ""     ),
        "output_path" : ( ["-o" ], "Path to output files"           , "dir?" ),
        "catalog_path": ( ["-l" ], "Path to catalog files"          , "dir"  ),
    },
)
add_command(
    haloutils.correlation2.abacus_corrfunc,
    help    = "Estimate correlation function from abacus halo catalogs",
    version = "",
    options = {
        "simname" : ( ["-s" ], "Name of simulation"               , ""     ),
        "redshift": ( ["-z" ], "Redshift value"                   , ""     ),
        "rbins"   : ( ["-r" ], "Distance bin edges (Mpc)"         , ""     ),
        "mrange1" : ( ["-m1"], "Mass range for first set (Msun)"  , ""     ),
        "mrange2" : ( ["-m2"], "Mass range for second set (Msun)" , ""     ),
        "outfile" : ( ["-o" ], "Path to output file"              , ""     ), 
        "nthreads": ( ["-n" ], "Number of threads to use"         , ""     ),
        "subdivs" : ( ["-j" ], "Number of jackknife samples"      , ""     ),
        "rseed"   : ( ["-rs"], "Random seed"                      , ""     ),
        "randsize": ( ["-f" ], "Random catalog size"              , ""     ),
        "workdir" : ( ["-w" ], "Working directory"                , "dir?" ), 
        "loc"     : ( ["-l" ], "Path to catalog files"            , "dir"  ),
    },
)
add_command(
    haloutils.correlation2.galaxy_corrfunc, 
    help    = "Estimate correlation function from galaxy catalogs",
    version = "",
    options = {
        "loc"     : ( ["-l" ], "Path to catalog files"           , ""     ),
        "rbins"   : ( ["-r" ], "Distance bin edges (Mpc)"        , ""     ),
        "mrange1" : ( ["-m1"], "Mass range for first set (Msun)" , ""     ),
        "mrange2" : ( ["-m2"], "Mass range for second set (Msun)", ""     ),
        "outfile" : ( ["-o" ], "Path to output file"             , ""     ), 
        "nthreads": ( ["-n" ], "Number of threads to use"        , ""     ),
        "subdivs" : ( ["-j" ], "Number of jackknife samples"     , ""     ),
        "rseed"   : ( ["-rs"], "Random seed"                     , ""     ),
        "randsize": ( ["-f" ], "Random catalog size"             , ""     ),
        "workdir" : ( ["-w" ], "Working directory"               , "dir?" ), 
    },
)
add_command(
    haloutils.mass_function.abacus_massfunction,
    help    = "Estimate halo mass-function from abacus halo catalogs",
    version = "",
    options = {
        "simname" : ( ["-s"], "Name of the abacus simulation", ""    ),
        "redshift": ( ["-z"], "Redshift of simulation"       , ""    ),
        "bins"    : ( ["-b"], "Mass bin edges in Msun"       , ""    ),
        "outfile" : ( ["-o"], "Path for the output file"     , ""    ),
        "nprocs"  : ( ["-n"], "Number of threads"            , ""    ),
        "smooth"  : ( ["-f"], "Size of the smoothing window" , ""    ),
        "loc"     : ( ["-l"], "Path to search halo catalogs" , "dir" ),
    },
)

if __name__ == "__main__": cli()
