import logging
import click
import os
import sys
import pandas as pd
from typing import Optional, Tuple

# --- Configure Logging ---
# Basic config here, gets potentially overridden by config file later in load_config
logging.basicConfig(
    level=logging.INFO, # Default level
    format='%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)] # Log to stdout
)
# Silence overly verbose libraries by default
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("numexpr").setLevel(logging.WARNING)

logger = logging.getLogger("drwiggle.cli") # Logger specific to CLI

# --- Import Core Components ---
# Defer heavy imports until commands are run if possible
# (though pipeline import might trigger others)
from drwiggle.config import load_config
from drwiggle.pipeline import Pipeline


# --- Helper Functions ---
def _setup_pipeline(ctx, config_path: Optional[str], param_overrides: Optional[Tuple[str]], cli_option_overrides: dict) -> Pipeline:
    """Loads config and initializes the pipeline."""
    try:
        # Pass CLI overrides directly to load_config
        # Resolve paths relative to the current working directory (where CLI is run)
        cfg = load_config(
            config_path=config_path,
            cli_overrides=cli_option_overrides,
            param_overrides=param_overrides,
            resolve_paths_base_dir=os.getcwd() # Resolve relative to CWD
        )
        # Store config in context for potential use by other commands if needed
        ctx.obj = cfg
        pipeline = Pipeline(cfg)
        return pipeline
    except FileNotFoundError as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)
    except (ValueError, TypeError, KeyError) as e:
         logger.error(f"Configuration Error: Invalid setting or structure - {e}", exc_info=True)
         sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
        sys.exit(1)

# --- Click CLI Definition ---

@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.version_option(version="1.0.0", package_name="drwiggle") # Assumes setup.py version
@click.option('--config', '-c', type=click.Path(exists=True, dir_okay=False), help='Path to custom YAML config file.')
@click.option('--param', '-p', multiple=True, help='Override config param (key.subkey=value). Can be used multiple times.')
@click.pass_context # Pass context to store config
def cli(ctx, config, param):
    """
    drWiggle: Protein Flexibility Classification Framework.

    Train models, evaluate performance, predict flexibility, and analyze results
    across different temperatures based on RMSF data and structural features.

    Configuration is loaded from default_config.yaml, overridden by the --config file,
    environment variables (DRWIGGLE_*), and finally --param options.
    """
    # Store base config path and params in context for commands to use
    # The actual config loading happens within each command using _setup_pipeline
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['param_overrides'] = param
    logger.info("drWiggle CLI started.")


@cli.command()
@click.option("--model", '-m', help="Model(s) to train (comma-separated, e.g., 'random_forest,neural_network'). Default: all enabled in config.")
@click.option("--input", '-i', type=click.Path(resolve_path=True), help="Input data file/pattern. Overrides 'dataset.file_pattern' in config.")
@click.option("--temperature", '-t', type=str, help="Temperature context (e.g., 320). Overrides 'temperature.current'. REQUIRED if data pattern uses {temperature}.")
@click.option("--binning", '-b', type=click.Choice(["kmeans", "quantile"], case_sensitive=False), help="Override binning method.")
@click.option("--output-dir", '-o', type=click.Path(resolve_path=True), help="Override 'paths.output_dir'.")
@click.option("--models-dir", type=click.Path(resolve_path=True), help="Override 'paths.models_dir'.")
@click.pass_context
def train(ctx, model, input, temperature, binning, output_dir, models_dir):
    """Train flexibility classification model(s)."""
    logger.info("=== Train Command Initiated ===")
    # Prepare CLI overrides dictionary for load_config
    cli_overrides = {}
    if temperature: cli_overrides.setdefault('temperature', {})['current'] = temperature
    if binning: cli_overrides.setdefault('binning', {})['method'] = binning
    if output_dir: cli_overrides.setdefault('paths', {})['output_dir'] = output_dir
    if models_dir: cli_overrides.setdefault('paths', {})['models_dir'] = models_dir
    # Input override needs careful handling - pass directly to pipeline method
    # if input: cli_overrides.setdefault('dataset', {})['file_pattern'] = input # This isn't quite right, input can be path

    pipeline = _setup_pipeline(ctx, ctx.obj['config_path'], ctx.obj['param_overrides'], cli_overrides)

    # Temperature check: crucial if file_pattern uses {temperature} and --input not given
    if input is None and '{temperature}' in pipeline.config['dataset']['file_pattern']:
         current_temp = pipeline.config.get("temperature", {}).get("current")
         if current_temp is None:
              logger.error("Training data pattern requires {temperature}, but temperature not set via --temperature or config.")
              sys.exit(1)
         logger.info(f"Using temperature {current_temp} for finding training data.")


    model_list = model.split(',') if model else None # Pass None to train all enabled

    try:
        pipeline.train(model_names=model_list, data_path=input)
        logger.info("=== Train Command Finished Successfully ===")
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


@cli.command()
@click.option("--model", '-m', help="Model(s) to evaluate (comma-separated). Default: All models found in models_dir.")
@click.option("--input", '-i', type=click.Path(resolve_path=True), help="Evaluate on specific data file/pattern. Default: Use test split from training data source.")
@click.option("--temperature", '-t', type=str, help="Temperature context for loading models/data (e.g., 320). REQUIRED if default data pattern needs it.")
@click.option("--output-dir", '-o', type=click.Path(resolve_path=True), help="Override 'paths.output_dir'.")
@click.option("--models-dir", type=click.Path(resolve_path=True), help="Override 'paths.models_dir'.")
@click.pass_context
def evaluate(ctx, model, input, temperature, output_dir, models_dir):
    """Evaluate trained classification model(s)."""
    logger.info("=== Evaluate Command Initiated ===")
    cli_overrides = {}
    if temperature: cli_overrides.setdefault('temperature', {})['current'] = temperature
    if output_dir: cli_overrides.setdefault('paths', {})['output_dir'] = output_dir
    if models_dir: cli_overrides.setdefault('paths', {})['models_dir'] = models_dir

    pipeline = _setup_pipeline(ctx, ctx.obj['config_path'], ctx.obj['param_overrides'], cli_overrides)

    # Temperature check if default test set derivation needs it
    if input is None and '{temperature}' in pipeline.config['dataset']['file_pattern']:
         current_temp = pipeline.config.get("temperature", {}).get("current")
         if current_temp is None:
              logger.error("Deriving test set requires {temperature} in data pattern, but temperature not set via --temperature or config.")
              sys.exit(1)
         logger.info(f"Using temperature {current_temp} for deriving test set.")

    model_list = model.split(',') if model else None

    try:
        pipeline.evaluate(model_names=model_list, data_path=input)
        logger.info("=== Evaluate Command Finished Successfully ===")
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


@cli.command()
@click.option("--input", '-i', type=click.Path(exists=True, dir_okay=False, resolve_path=True), required=True, help="Input data file (CSV) for prediction.")
@click.option("--model", '-m', type=str, help="Model name to use. Default: 'random_forest'.")
@click.option("--output", '-o', type=click.Path(resolve_path=True), help="Output file path for predictions (CSV). Default: derive from input filename.")
@click.option("--temperature", '-t', type=str, help="Temperature context for loading model (e.g., 320). Sets 'temperature.current' in config.")
@click.option("--probabilities", is_flag=True, default=False, help="Include class probabilities in output.")
@click.option("--models-dir", type=click.Path(resolve_path=True), help="Override 'paths.models_dir'.")
@click.pass_context
def predict(ctx, input, model, output, temperature, probabilities, models_dir):
    """Predict flexibility classes for new data."""
    logger.info("=== Predict Command Initiated ===")
    cli_overrides = {}
    if temperature: cli_overrides.setdefault('temperature', {})['current'] = temperature
    if models_dir: cli_overrides.setdefault('paths', {})['models_dir'] = models_dir
    # Store probability flag for pipeline to access
    cli_overrides.setdefault('cli_options', {})['predict_probabilities'] = probabilities

    pipeline = _setup_pipeline(ctx, ctx.obj['config_path'], ctx.obj['param_overrides'], cli_overrides)

    if not output:
        base, ext = os.path.splitext(input)
        output = f"{base}_predictions.csv"
        logger.info(f"Output path not specified, defaulting to: {output}")

    try:
        predictions_df = pipeline.predict(data=input, model_name=model, output_path=output)
        # Predict method handles saving if output_path is given
        if predictions_df is not None:
             # This happens if output_path wasn't specified or saving failed
             logger.info("Prediction method returned DataFrame (likely because output_path was None or saving failed).")
        logger.info("=== Predict Command Finished Successfully ===")
    except Exception as e:
        logger.error(f"Prediction pipeline failed: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


@cli.command()
@click.option("--pdb", required=True, help="PDB ID (e.g., '1AKE') or path to a local PDB file.")
@click.option("--model", '-m', type=str, help="Model name to use for prediction. Default: 'random_forest'.")
@click.option("--temperature", '-t', type=str, help="Temperature context for prediction model (e.g., 320). Sets 'temperature.current'.")
@click.option("--output-prefix", '-o', type=click.Path(resolve_path=True), help="Output prefix for generated files (e.g., ./output/1ake_flex). Default: '{output_dir}/pdb_vis/{pdb_id}_{model}_flexibility'")
@click.option("--models-dir", type=click.Path(resolve_path=True), help="Override 'paths.models_dir'.")
@click.option("--pdb-cache-dir", type=click.Path(resolve_path=True), help="Override 'paths.pdb_cache_dir'.")
@click.pass_context
def process_pdb(ctx, pdb, model, temperature, output_prefix, models_dir, pdb_cache_dir):
    """Fetch/Parse PDB, Extract Features, Predict Flexibility, and Generate Visualizations."""
    logger.info("=== Process PDB Command Initiated ===")
    cli_overrides = {}
    if temperature: cli_overrides.setdefault('temperature', {})['current'] = temperature
    if models_dir: cli_overrides.setdefault('paths', {})['models_dir'] = models_dir
    if pdb_cache_dir: cli_overrides.setdefault('paths', {})['pdb_cache_dir'] = pdb_cache_dir
    # Ensure PDB processing is enabled in the loaded config
    cli_overrides.setdefault('pdb', {})['enabled'] = True

    pipeline = _setup_pipeline(ctx, ctx.obj['config_path'], ctx.obj['param_overrides'], cli_overrides)

    try:
        pipeline.process_pdb(pdb_input=pdb, model_name=model, output_prefix=output_prefix)
        logger.info("=== Process PDB Command Finished Successfully ===")
    except Exception as e:
        logger.error(f"PDB processing pipeline failed: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


@cli.command()
@click.option("--input", '-i', type=click.Path(exists=True, dir_okay=False, resolve_path=True), required=True, help="Input RMSF data file (CSV) for analysis.")
@click.option("--temperature", '-t', type=str, help="Temperature context (e.g., 320).")
@click.option("--output-dir", '-o', type=click.Path(resolve_path=True), help="Directory to save the plot. Overrides 'paths.output_dir'.")
@click.option("--models-dir", type=click.Path(resolve_path=True), help="Directory containing saved binner. Overrides 'paths.models_dir'.")
@click.pass_context
def analyze_distribution(ctx, input, temperature, output_dir, models_dir):
    """Analyze RMSF distribution and visualize binning boundaries."""
    logger.info("=== Analyze Distribution Command Initiated ===")
    cli_overrides = {}
    if temperature: cli_overrides.setdefault('temperature', {})['current'] = temperature
    if output_dir: cli_overrides.setdefault('paths', {})['output_dir'] = output_dir
    if models_dir: cli_overrides.setdefault('paths', {})['models_dir'] = models_dir

    pipeline = _setup_pipeline(ctx, ctx.obj['config_path'], ctx.obj['param_overrides'], cli_overrides)

    plot_filename = f"rmsf_distribution_analysis_{temperature or 'default'}.png"
    plot_path = os.path.join(pipeline.config['paths']['output_dir'], plot_filename)

    try:
        pipeline.analyze_rmsf_distribution(input_data_path=input, output_plot_path=plot_path)
        logger.info("=== Analyze Distribution Command Finished Successfully ===")
    except Exception as e:
        logger.error(f"RMSF distribution analysis failed: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


@cli.command()
@click.option("--model", '-m', type=str, help="Model name to focus comparison on (optional).")
@click.option("--output-dir", '-o', type=click.Path(resolve_path=True), help="Override base 'paths.output_dir' for finding results and saving comparison.")
@click.pass_context
def compare_temperatures(ctx, model, output_dir):
    """Compare classification results across different temperatures."""
    logger.info("=== Compare Temperatures Command Initiated ===")
    cli_overrides = {}
    # Output dir override applies to the base dir where temp results are sought
    if output_dir: cli_overrides.setdefault('paths', {})['output_dir'] = output_dir
    # Temperature override doesn't make sense here as we compare multiple temps

    pipeline = _setup_pipeline(ctx, ctx.obj['config_path'], ctx.obj['param_overrides'], cli_overrides)

    try:
        pipeline.run_temperature_comparison(model_name=model)
        logger.info("=== Compare Temperatures Command Finished Successfully ===")
    except Exception as e:
        logger.error(f"Temperature comparison failed: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


@cli.command()
@click.option("--predictions", type=click.Path(exists=True, dir_okay=False, resolve_path=True), required=True, help="Path to the predictions CSV file (must contain 'predicted_class').")
@click.option("--output-dir", '-o', type=click.Path(resolve_path=True), help="Directory to save visualizations. Overrides 'paths.output_dir'.")
@click.pass_context
def visualize(ctx, predictions, output_dir):
    """Generate visualizations from saved prediction files."""
    logger.info("=== Visualize Command Initiated ===")
    cli_overrides = {}
    if output_dir: cli_overrides.setdefault('paths', {})['output_dir'] = output_dir
    # Temperature override might be needed if config depends on it for vis settings
    # Add --temperature option if necessary later.

    pipeline = _setup_pipeline(ctx, ctx.obj['config_path'], ctx.obj['param_overrides'], cli_overrides)

    try:
        pipeline.visualize_results(predictions_path=predictions, output_dir=output_dir) # Pass specified output dir
        logger.info("=== Visualize Command Finished Successfully ===")
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


# Main entry point for script execution
if __name__ == '__main__':
    # Set process title if possible (useful for monitoring)
    try:
        import setproctitle
        setproctitle.setproctitle("drwiggle_cli")
    except ImportError:
        pass
    # Execute the Click application
    cli()
