"""Main script for knowledge distillation experiments."""

import os
import argparse
import torch

from config import *
from data_utils import load_ucr_dataset, prepare_data, create_data_loaders
from trainer import Trainer
from utils import (
    save_logs, get_best_teacher, create_directory, 
    check_done, mark_done, print_model_summary
)


def run_experiment(dataset_name, classifier_name, iteration, archive_name,
                   alpha=None, temperature=None):
    """
    Run a single experiment.
    
    Args:
        dataset_name: Name of the dataset
        classifier_name: Type of classifier ('teacher', 'student_kd', 'student_alone')
        iteration: Iteration number
        archive_name: Archive name
        alpha: Alpha parameter for KD (if applicable)
        temperature: Temperature for KD (if applicable)
    """
    # Determine output directory
    if classifier_name == 'student_kd':
        output_dir = os.path.join(
            PATH_OUT, 'results', classifier_name,
            f'alpha_{alpha}', f'temperature_{temperature}',
            f'{archive_name}_itr_{iteration}', dataset_name
        )
    else:
        output_dir = os.path.join(
            PATH_OUT, 'results', classifier_name,
            f'{archive_name}_itr_{iteration}', dataset_name
        )
    
    # Check if already completed
    if check_done(output_dir):
        print(f"\nExperiment already completed: {output_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Starting experiment:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Classifier: {classifier_name}")
    print(f"  Iteration: {iteration}")
    if classifier_name == 'student_kd':
        print(f"  Alpha: {alpha}")
        print(f"  Temperature: {temperature}")
    print(f"  Output: {output_dir}")
    print(f"{'='*80}\n")
    
    # Create output directory
    create_directory(output_dir)
    
    # Load dataset
    print(f"Loading dataset {dataset_name}...")
    x_train, y_train, x_test, y_test = load_ucr_dataset(PATH_DATA, dataset_name)
    
    # Prepare data
    x_train, y_train, x_test, y_test, nb_classes = prepare_data(
        x_train, y_train, x_test, y_test, one_hot=True
    )
    
    print(f"Data shapes:")
    print(f"  Train: {x_train.shape}, Classes: {nb_classes}")
    print(f"  Test: {x_test.shape}")
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        x_train, y_train, x_test, y_test,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Determine input shape
    input_shape = x_train.shape[1:]  # (seq_length, n_features)
    
    # Get teacher path if needed
    teacher_path = None
    if classifier_name == 'student_kd':
        teacher_root = os.path.join(PATH_OUT, 'results', 'teacher')
        teacher_path = get_best_teacher(dataset_name, teacher_root, 
                                       num_iterations=ITERATIONS['teacher'])
    
    # Create trainer
    if classifier_name == 'teacher':
        model_depth = TEACHER_DEPTH
    else:
        model_depth = STUDENT_DEPTH
    
    trainer = Trainer(
        model_type=classifier_name,
        input_shape=input_shape,
        nb_classes=nb_classes,
        output_dir=output_dir,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        patience=PATIENCE,
        min_lr=MIN_LR,
        lr_factor=LR_FACTOR,
        device=DEVICE,
        teacher_path=teacher_path,
        alpha=alpha if alpha is not None else 0.3,
        temperature=temperature if temperature is not None else 10.0,
        teacher_depth=TEACHER_DEPTH,
        student_depth=STUDENT_DEPTH,
        nb_filters=NB_FILTERS
    )
    
    # Print model summary
    print_model_summary(trainer.model, input_shape)
    
    # Train model
    duration = trainer.fit(train_loader)
    
    # Save logs and metrics
    save_logs(trainer.model, test_loader, output_dir, 
             trainer.history, duration, device=DEVICE)
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'last_model.pth')
    trainer.save_model(final_model_path)
    
    # Mark as done
    mark_done(output_dir)
    
    print(f"\nExperiment completed: {output_dir}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    """Main function to run all experiments."""
    parser = argparse.ArgumentParser(description='Knowledge Distillation Experiments')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific datasets to run (default: all)')
    parser.add_argument('--classifiers', nargs='+', default=None,
                       help='Specific classifiers to run (default: all)')
    parser.add_argument('--iterations', type=int, default=None,
                       help='Number of iterations (default: from config)')
    args = parser.parse_args()
    
    # Determine which datasets and classifiers to run
    datasets_to_run = args.datasets if args.datasets else UNIVARIATE_DATASET_NAMES_2018
    classifiers_to_run = args.classifiers if args.classifiers else CLASSIFIERS
    
    print("\n" + "="*80)
    print("Knowledge Distillation Experiments - PyTorch Implementation")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Datasets: {len(datasets_to_run)}")
    print(f"Classifiers: {classifiers_to_run}")
    print(f"Output directory: {PATH_OUT}")
    print("="*80 + "\n")
    
    # Run experiments
    for archive_name in ARCHIVE_NAMES:
        for dataset_name in datasets_to_run:
            for classifier_name in classifiers_to_run:
                # Determine number of iterations
                if args.iterations:
                    num_iters = args.iterations
                else:
                    num_iters = ITERATIONS.get(classifier_name, 5)
                
                for iteration in range(1, num_iters + 1):
                    if classifier_name == 'student_kd':
                        # Run with different alpha and temperature values
                        for alpha in ALPHA_LIST:
                            for temperature in TEMPERATURE_LIST:
                                try:
                                    run_experiment(
                                        dataset_name, classifier_name, iteration,
                                        archive_name, alpha, temperature
                                    )
                                except Exception as e:
                                    print(f"\nError in experiment: {e}")
                                    import traceback
                                    traceback.print_exc()
                    else:
                        # Run standard experiment
                        try:
                            run_experiment(
                                dataset_name, classifier_name, iteration,
                                archive_name
                            )
                        except Exception as e:
                            print(f"\nError in experiment: {e}")
                            import traceback
                            traceback.print_exc()
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()