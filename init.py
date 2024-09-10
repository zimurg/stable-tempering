import subprocess
import sys
import os
import argparse
from pathlib import Path


def check_and_install_requirements():
    marker_file = Path(".requirements_installed")

    if not marker_file.exists():
        print("Instalando las dependencias...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            marker_file.touch()
            print("Se han instalado las dependencias.")
        except subprocess.CalledProcessError as e:
            print(f"Se ha producido un error al instalar las dependencias: {e}")
            sys.exit(1)
    else:
        print("Todas las dependencias están instalados.")

def update_requirements():
    try:
        print("Actualizando las dependencias...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "-r", "requirements.txt"])
        print("Se han actualizado las dependencias.")
    except subprocess.CalledProcessError as e:
        print(f"Se ha producido un error al actualizar las dependencias: {e}")
        sys.exit(1)
    marker_file = Path(".requirements_installed")
    marker_file.touch()

def initialize_tempered_inference(prompts, seed=2024, steps=20, stop_time=15, backsteps=5, k=4, CFG_rescale=10):
    from loaders import SD_loader
    import stable_tempering



    pipeline=SD_loader()
    stable_tempering.stable_tempering(
        pipeline=pipeline,
        seed=seed,
        prompts=prompts,
        steps=steps,
        stop_time=stop_time,
        backsteps=backsteps,
        k=k,
        CFG_rescale=CFG_rescale
        )

def initialize_inference_test(prompts, seed=2024, steps=20, stop_time=15, backsteps=5, k=4, CFG_rescale=10, cycles=1, threshold=0.1):
    import tests


    tests.run_inference_test(prompts,
                             seed=seed,
                             steps=steps,
                             stop_time=stop_time,
                             backsteps=backsteps,
                             k=k,
                             CFG_rescale=CFG_rescale,
                             cycles=cycles,
                             threshold=threshold
                            )


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Tempered diffusion")
    parser.add_argument("--update", action="store_true", help="Actualizar las dependencias.")
    parser.add_argument("--tempered_inference", action="store_true", help="Ejecuta una pasada de inferencia temperada.")
    parser.add_argument("--evaluate", action="store_true", help="Evalúa la reducción del sangrado conceptual lograda en una inferencia temperada frente a una normal.")
    parser.add_argument("--cycles", type=int, default=1, help="Número de pasadas.")
    parser.add_argument("--prompts", type=str, nargs='+', help="Prompts para la generación. Cada sujeto debe introducirse entre comillas ej: \"a black dog\" \"a toy robot\".")
    parser.add_argument("--seed", type=int, default=2024, help="El valor semilla del generador.")
    parser.add_argument("--steps", type=int, default=20, help="Número de pasos de inferencia.")
    parser.add_argument("--stoptime", type=int, default=15, help="El paso en que se interrumpirá la inferencia.")
    parser.add_argument("--backsteps", type=int, default=5, help="Número de pasos de reinyección de ruido.")
    parser.add_argument("--k", type=float, default=4, help="Hiperparámetro que controla la forma de la distribución de la que se extrae el ruido (mayor valor: más dispersión).")
    parser.add_argument("--cfg_rescale", type=float, default=10, help="Escala el valor del ruido para adaptarlo a los valores de CFG.")

    check_and_install_requirements()
    args = parser.parse_args()

    if args.update:
        update_requirements()
        
    if args.tempered_inference:
        if not args.prompts:
            print("Error: Debe proporcionarse un prompt o lista de prompts.")
            sys.exit(1)
            
        initialize_tempered_inference(args.prompts,
                                    seed=args.seed,
                                    steps=args.steps,
                                    stop_time=args.stoptime,
                                    backsteps=args.backsteps,
                                    k=args.k,
                                    CFG_rescale=args.cfg_rescale
                                    )
        
    elif args.evaluate:
        if not args.prompts:
            print("Error: Debe proporcionarse un prompt o lista de prompts.")
            sys.exit(1)

        initialize_inference_test(args.prompts, 
                                  seed=args.seed,
                                  steps=args.steps,
                                  stop_time=args.stoptime,
                                  backsteps=args.backsteps,
                                  k=args.k, 
                                  CFG_rescale=args.cfg_rescale,
                                  cycles=args.cycles
                                 )
        
    else:
        print("Debe incluirse uno de los siguientes argumentos:\n--tempered_inference,\n--evaluate")
        parser.print_help()
