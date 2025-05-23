""" 
Module for handling predictions with torch 
""" 

from src import model_onehot
from src import format_data
import torch
import torch.nn.functional as F
import os
import numpy as np
from sklearn.neighbors import KernelDensity
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from loguru import logger
from torch.utils.data import DataLoader
import traceback

class Predictor: 

    def __init__(self, device = 'cuda'): 
        self.device = torch.device(device)
        self.models = []

    def predict_batch(self, embeddings, src_key_padding_mask):
        """
        Predict the scores for a batch of embeddings.

        :param embeddings: Tensor of input embeddings
        :param src_key_padding_mask: Mask tensor for padding in the input sequences
        :return: Dictionary of predicted scores
        """
        self.models = [model.to(self.device) for model in self.models]
        embeddings = embeddings.to(self.device)  # Move embeddings to device
        src_key_padding_mask = src_key_padding_mask.to(self.device)  # Move mask to device

        for model in self.models:
            model.eval()

        all_scores = {}

        with torch.no_grad():
            for model in self.models:
                outputs = model(embeddings, src_key_padding_mask=src_key_padding_mask)
                outputs = F.softmax(outputs, dim=-1)
                
                if len(all_scores) == 0:
                    all_scores = outputs.cpu().numpy()
                else:
                    all_scores += outputs.cpu().numpy()
        
        return all_scores   

    def predict(self, X, y): # not sure if this is right 
        """
        Predict the scores for the given input data.

        :param X: List of input data values
        :param y: List of target data values
        :return: Dictionary of predicted scores
        """
        self.models = [model.to(self.device) for model in self.models]
        for model in self.models:
            model.eval()

        all_scores = {}

        # Find the maximum length of the input sequences
        max_length = max([v.shape[0] for v in X])

        # Pad the input sequences to the same length
        X_padded = [F.pad(x, (0, 0, 0, max_length - x.shape[0])) for x in X]

        # Convert input data to tensors
        X_tensor = torch.stack(X_padded).to(self.device)

        # Compute src_key_padding using format_data.pad_sequence
        src_key_padding = format_data.pad_sequence(y)
        src_key_padding_tensor = torch.tensor(src_key_padding).to(self.device)

        with torch.no_grad():
            for model in self.models:
                outputs = model(X_tensor, src_key_padding_mask=src_key_padding_tensor)
                outputs = F.softmax(outputs, dim=-1)
                for i, key in enumerate(y.keys()):
                    if key not in all_scores:
                        all_scores[key] = outputs[i].cpu().numpy()
                    else:
                        all_scores[key] += outputs[i].cpu().numpy()

        self.scores = all_scores
        return all_scores  # Remove the TODO comment if the return statement is necessary
    
    def predict_inference(self, X, y, batch_size=128):
        """
        Predict using batch processing for inference.
        
        :param X: Dictionary of embeddings with keys as identifiers
        :param y: Dictionary of categories with keys as identifiers
        :param batch_size: Size of batches for processing
        :return: Dictionary of predicted scores with keys matching input dictionaries
        """
        logger.info(f"Starting batch inference with batch size {batch_size}")
        
        # Prepare for batch processing
        self.models = [model.to(self.device) for model in self.models]
        for model in self.models:
            model.eval()
        
        # Create dataset
        keys_list = list(X.keys())
        dataset = model_onehot.EmbeddingDataset(
            list(X.values()), 
            list(y.values()), 
            keys_list,
            mask_portion=0  # No masking for inference
        )
        
        # Set inference mode
        dataset.training = False
        dataset.validation = False
        
        # Create DataLoader
        data_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            collate_fn=model_onehot.collate_fn
        )
        
        logger.info(f"Processing {len(keys_list)} samples in batches of {batch_size}")
        
        # Process batches
        all_scores = {}
        batches_processed = 0
        total_samples_processed = 0
        
        # Track which indices we've processed for debugging
        processed_indices = set()
        
        for batch_idx, (embeddings_batch, categories_batch, masks_batch, idx_batch) in enumerate(data_loader):
            try:
                # Move tensors to device
                embeddings_batch = embeddings_batch.to(self.device)
                masks_batch = masks_batch.to(self.device)
                src_key_padding_mask = (masks_batch != -2).to(self.device)
                
                # Get predictions for this batch
                logger.debug(f"Processing batch {batch_idx} with {len(idx_batch)} items")
                batch_scores = self.predict_batch(embeddings_batch, src_key_padding_mask)
                
                # Store scores by their original keys
                # Calculate the absolute index in the dataset for this batch
                batch_start_idx = batch_idx * batch_size
                
                # This is the critical part: map batch positions to original keys properly
                for i in range(len(idx_batch)):
                    absolute_idx = batch_start_idx + i
                    if absolute_idx < len(keys_list):
                        key = keys_list[absolute_idx]
                        processed_indices.add(absolute_idx)
                        
                        if isinstance(batch_scores, torch.Tensor):
                            if i < batch_scores.shape[0]:
                                all_scores[key] = batch_scores[i].cpu().numpy()
                                total_samples_processed += 1
                        else:
                            # It's already a numpy array
                            if i < len(batch_scores):
                                all_scores[key] = batch_scores[i]
                                total_samples_processed += 1
                
                batches_processed += 1
                logger.debug(f"Processed batch {batches_processed} with {len(idx_batch)} items (total: {total_samples_processed}/{len(keys_list)} samples)")
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Check if we processed all genomes
        if len(all_scores) < len(keys_list):
            missing = len(keys_list) - len(all_scores)
            logger.warning(f"Missing scores for {missing} genomes after processing all batches")
            
            # Find missing keys
            processed_keys = set(all_scores.keys())
            all_keys = set(keys_list)
            missing_keys = all_keys - processed_keys
            logger.warning(f"Missing keys: {list(missing_keys)[:10]}...")
        
        logger.info(f"Total batches processed: {batches_processed}")
        logger.info(f"Collected scores for {len(all_scores)}/{len(keys_list)} samples")
        
        # Store scores for compatibility with other methods
        self.scores = all_scores
        return all_scores

    def write_predictions_to_genbank(self, gb_dict, output_file, predictions, scores, confidence_scores, threshold=0.9, categories_map=None):
        """
        Write the predictions to a GenBank file, creating one record per genome with sequences as features.
        Only add Phynteny predictions when function is unknown or missing.
        
        :param gb_dict: Dictionary containing genome information with nested sequence records
        :param output_file: Path to the output GenBank file
        :param predictions: List of predictions for each sequence
        :param scores: List of scores for each prediction
        :param confidence_scores: List of confidence scores for each prediction
        :param threshold: Confidence threshold (default: 0.9)
        :param categories_map: Dictionary mapping category numbers to human-readable labels
        :return: None
        """
        logger.info(f"Writing predictions to GenBank file: {output_file}")
        
        # Track statistics for diagnostics
        unknown_function_count = 0
        predictions_added_count = 0
        low_confidence_count = 0
        known_function_count = 0
        
        # Log the dimensions of the data arrays
        logger.debug(f"Number of genomes in gb_dict: {len(gb_dict)}")
        logger.debug(f"Number of prediction arrays: {len(predictions) if predictions is not None else 'None'}")
        logger.debug(f"Number of score arrays: {len(scores) if scores is not None else 'None'}")
        logger.debug(f"Number of confidence arrays: {len(confidence_scores) if confidence_scores is not None else 'None'}")
        
        # Log the first few genome IDs to verify alignment
        genome_ids = list(gb_dict.keys())
        logger.info(f"First 5 genome IDs: {genome_ids[:5] if len(genome_ids) >= 5 else genome_ids}")
        
        try:
            with open(output_file, "w") as handle:
                # For each genome in the gb_dict
                for genome_idx, (genome_id, genome_data) in enumerate(gb_dict.items()):
                    logger.info(f"Processing genome #{genome_idx}: {genome_id}")
                    
                    if 'sequence' not in genome_data:
                        logger.warning(f"No sequence field found for genome {genome_id}")
                        continue
                    
                    sequences = genome_data.get('sequence', [])
                    
                    if not sequences:
                        logger.warning(f"No sequences found for genome {genome_id}")
                        continue
                    
                    logger.info(f"Genome {genome_id} has {len(sequences)} sequences")
                    
                    # Check if we have predictions for this genome index
                    has_predictions = predictions is not None and genome_idx < len(predictions)
                    logger.debug(f"Has predictions for genome {genome_id}: {has_predictions}")
                    if has_predictions:
                        logger.debug(f"Number of predictions for genome {genome_id}: {len(predictions[genome_idx])}")
                    
                    # Check if we have scores for this genome index
                    has_scores = scores is not None and genome_idx < len(scores)
                    logger.debug(f"Has scores for genome {genome_id}: {has_scores}")
                    if has_scores:
                        logger.debug(f"Number of scores for genome {genome_id}: {len(scores[genome_idx])}")
                    
                    # Check if we have confidence scores for this genome index
                    has_confidence = confidence_scores is not None and genome_idx < len(confidence_scores)
                    logger.debug(f"Has confidence scores for genome {genome_id}: {has_confidence}")
                    
                    # Create a new record for this genome
                    # Use the first sequence record to extract metadata if available
                    sample_record = sequences[0]
                    
                    # Get length information if available
                    genome_length = genome_data.get('length', 0)
                    if genome_length == 0:
                        # Calculate total length from positions if available
                        if 'position' in genome_data:
                            positions = genome_data['position']
                            if positions:
                                # Find the furthest endpoint
                                genome_length = max([pos[1] for pos in positions])
                    
                    # Use the original genome sequence if available, otherwise use placeholder
                    genome_seq = None
                    if 'original_sequence' in genome_data and genome_data['original_sequence']:
                        genome_seq = Seq(genome_data['original_sequence'])
                    else:
                        # Fallback to placeholder sequence
                        genome_seq = Seq('N' * genome_length) if genome_length > 0 else Seq('')
                        logger.warning(f"Using placeholder sequence for genome {genome_id} - original sequence not found")
                    
                    # Create the genome record
                    genome_record = SeqRecord(
                        seq=genome_seq,
                        id=genome_id,
                        name=genome_id,
                        description=f"{genome_id} - Phynteny annotated genome"
                    )
                    
                    # Set required molecule_type
                    genome_record.annotations["molecule_type"] = "DNA"
                    
                    # Copy any other useful annotations from the first record if available
                    if sequences and hasattr(sequences[0], 'annotations'):
                        # List of annotations to preserve at the genome level
                        preserve_annotations = ['taxonomy', 'organism', 'source', 'date', 'accessions']
                        
                        for ann in preserve_annotations:
                            if ann in sequences[0].annotations:
                                genome_record.annotations[ann] = sequences[0].annotations[ann]
                    
                    # Process each sequence/gene in this genome
                    for seq_idx, record in enumerate(sequences):
                        try:
                            logger.debug(f"Processing sequence #{seq_idx} in genome {genome_id}")
                            
                            # Get prediction and confidence for this sequence
                            prediction = None
                            max_score = None
                            conf = None
                            
                            # Check array bounds before accessing
                            if has_predictions and seq_idx < len(predictions[genome_idx]):
                                prediction = int(predictions[genome_idx][seq_idx])
                                logger.debug(f"Genome {genome_id}, sequence {seq_idx}: Prediction = {prediction}")
                                
                                # Extract max score (at predicted category)
                                if has_scores and seq_idx < len(scores[genome_idx]):
                                    score_array = scores[genome_idx][seq_idx]
                                    logger.debug(f"Genome {genome_id}, sequence {seq_idx}: Score array shape = {score_array.shape if hasattr(score_array, 'shape') else 'scalar'}")
                                    
                                    if isinstance(score_array, np.ndarray) and len(score_array) > prediction:
                                        max_score = float(score_array[prediction])
                                        logger.debug(f"Genome {genome_id}, sequence {seq_idx}: Max score = {max_score}")
                                    else:
                                        max_score = score_array
                                        logger.debug(f"Genome {genome_id}, sequence {seq_idx}: Score = {max_score}")
                                else:
                                    logger.warning(f"Score array missing or index out of bounds for genome {genome_id}, sequence {seq_idx}")
                                
                                # Get confidence score
                                if has_confidence:
                                    if isinstance(confidence_scores[genome_idx], np.ndarray):
                                        if seq_idx < len(confidence_scores[genome_idx]):
                                            conf = confidence_scores[genome_idx][seq_idx]
                                            logger.debug(f"Genome {genome_id}, sequence {seq_idx}: Confidence = {conf}")
                                        else:
                                            logger.warning(f"Confidence index out of bounds: seq_idx {seq_idx} >= len(confidence_scores[genome_idx]) {len(confidence_scores[genome_idx])}")
                                    else:
                                        conf = confidence_scores[genome_idx]
                                        logger.debug(f"Genome {genome_id}: Genome-level confidence = {conf}")
                            else:
                                logger.warning(f"Prediction missing or index out of bounds for genome {genome_id}, sequence {seq_idx}")
                                if has_predictions:
                                    logger.warning(f"Array bounds issue: seq_idx={seq_idx}, predictions array length={len(predictions[genome_idx])}")
                            
                            # Get position information
                            start = 0
                            end = len(record.seq)
                            if 'position' in genome_data and seq_idx < len(genome_data['position']):
                                start, end = genome_data['position'][seq_idx]
                            
                            # Get strand information
                            strand = 1  # Default to forward strand
                            if 'sense' in genome_data and seq_idx < len(genome_data['sense']):
                                strand = 1 if genome_data['sense'][seq_idx] == '+' else -1
                            
                            # Start with an empty set of qualifiers
                            qualifiers = {}
                            
                            # Try to get qualifiers from the all_qualifiers field first (most comprehensive)
                            if 'all_qualifiers' in genome_data and seq_idx < len(genome_data['all_qualifiers']):
                                qualifiers = genome_data['all_qualifiers'][seq_idx].copy()
                            
                            # Next, check features (fallback)
                            elif hasattr(record, 'features') and record.features:
                                for feature in record.features:
                                    if feature.type == "CDS":
                                        # Deep copy ALL qualifiers from the original feature
                                        qualifiers = {key: value[:] if isinstance(value, list) else [value] 
                                                     for key, value in feature.qualifiers.items()}
                                        break
                            else:
                                logger.warning(f"No qualifiers found for {record.id} - creating basic set")
                            
                            # Ensure we have the essential qualifiers
                            if 'translation' not in qualifiers:
                                qualifiers['translation'] = [str(record.seq)]
                            
                            # Add product qualifier if not present
                            if 'product' not in qualifiers:
                                qualifiers['product'] = [f"protein_{seq_idx}"]
                            
                            # Add protein_id if available and not present
                            if 'protein_id' not in qualifiers and hasattr(record, 'id'):
                                qualifiers['protein_id'] = [record.id]
                            
                            # Check if we should add Phynteny predictions
                            should_add_predictions = True
                            
                            # Check if this is a gene with unknown function
                            has_unknown_function = False
                            if 'function' not in qualifiers:
                                has_unknown_function = True
                                unknown_function_count += 1
                                logger.debug(f"Gene {record.id} has no function annotation")
                            else:
                                function_value = qualifiers['function'][0].lower() if isinstance(qualifiers['function'], list) else qualifiers['function'].lower()
                                if function_value == "unknown function" or function_value == "unknown":
                                    has_unknown_function = True
                                    unknown_function_count += 1
                                    logger.debug(f"Gene {record.id} has unknown function: {function_value}")
                                else:
                                    known_function_count += 1
                                    should_add_predictions = False
                                    logger.debug(f"Skipping Phynteny annotation for {record.id}: Known function: {function_value}")

                            # Only add predictions if function is unknown or not present
                            if 'function' in qualifiers:
                                function_value = qualifiers['function'][0].lower() if isinstance(qualifiers['function'], list) else qualifiers['function'].lower()
                                if function_value != "unknown function" and function_value != "unknown":
                                    should_add_predictions = False
                                    logger.debug(f"Skipping Phynteny annotation for {record.id}: Known function: {function_value}")
                            
                            # Only add predictions if the confidence score is above the threshold
                            if conf is not None and conf < threshold:
                                should_add_predictions = False
                                low_confidence_count += 1
                                logger.debug(f"Skipping Phynteny annotation for {record.id}: Low confidence score: {conf:.4f}")
                            
                            # Add our prediction qualifiers only for unknown function
                            if should_add_predictions:
                                logger.debug(f'prediction: {prediction}')
                                if prediction is not None:
                                    # Add the category label if categories_map is provided
                                    if categories_map is not None:
                                        # First try direct lookup
                                        if prediction in categories_map:
                                            qualifiers["phynteny_category"] = [categories_map[prediction]]
                                            predictions_added_count += 1
                                            logger.debug(f"Added prediction for {record.id}: {categories_map[prediction]}")
                                        # If direct lookup fails, try string versions of keys
                                        elif str(prediction) in categories_map:
                                            qualifiers["phynteny_category"] = [categories_map[str(prediction)]]
                                            predictions_added_count += 1
                                            logger.debug(f"Added prediction for {record.id} via string key: {categories_map[str(prediction)]}")
                                        else:
                                            # Fallback to just showing the prediction number
                                            qualifiers["phynteny_category"] = [f"Category_{prediction}"]
                                            predictions_added_count += 1
                                            logger.debug(f"Added generic category for {record.id}: Category_{prediction}")
                                    else:
                                        # No categories_map provided
                                        qualifiers["phynteny_category"] = [f"Category_{prediction}"]
                                        predictions_added_count += 1
                                        
                                if max_score is not None:
                                    qualifiers["phynteny_score"] = [f"{max_score:.4f}"]
                                if conf is not None:
                                    qualifiers["phynteny_confidence"] = [f"{conf:.4f}"]
                            
                            # Create the feature with ALL original qualifiers plus our additions
                            feature = SeqFeature(
                                FeatureLocation(start, end, strand=strand),
                                type="CDS",
                                qualifiers=qualifiers
                            )
                            
                            genome_record.features.append(feature)
                            
                        except Exception as e:
                            logger.error(f"Error processing sequence {seq_idx} in genome {genome_id}: {str(e)}")
                            logger.error(traceback.format_exc())
                    
                    # Write the complete genome record with all features
                    SeqIO.write(genome_record, handle, "genbank")
                
                logger.info(f"Successfully wrote predictions to {output_file}")
                logger.info(f"Statistics: {unknown_function_count} genes with unknown function, {predictions_added_count} predictions added")
                logger.info(f"Skipped: {low_confidence_count} due to low confidence, {known_function_count} with known function")
                    
            return None
            
        except Exception as e:
            logger.error(f"Error writing to GenBank file {output_file}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def write_genbank(self, gb_dict, out, predictions=None, scores=None, confidence_scores=None, threshold=0.9, categories_map=None):
        """
        Write the predicted scores to Genbank files.

        :param gb_dict: Dictionary of Genbank records
        :param out: Output directory or output file path
        :param predictions: List of predictions (optional)
        :param scores: List of scores (optional)
        :param confidence_scores: List of confidence scores (optional)
        :param threshold: Confidence threshold (default: 0.9)
        :param categories_map: Dictionary mapping category numbers to human-readable labels
        :return: None
        """
        return self.write_predictions_to_genbank(gb_dict, out, predictions, scores, confidence_scores, threshold, categories_map)

    def read_model(self, model_path, input_dim, num_classes, num_heads, hidden_dim, lstm_hidden_dim, dropout, use_lstm, max_len, protein_dropout_rate=0.0, attention='circular', positional_encoding_type='fourier', pre_norm=False, progressive_dropout=False, initial_dropout_rate=1.0, final_dropout_rate=0.4, output_dim=None, num_layers=2): 
        """
        Read and load a model from the given path.

        :param model_path: Path to the model file
        :param input_dim: Input dimension for the model
        :param num_classes: Number of classes for the model
        :param num_heads: Number of attention heads for the model
        :param hidden_dim: Hidden dimension for the model
        :param lstm_hidden_dim: LSTM hidden dimension for the model
        :param dropout: Dropout rate for the model
        :param use_lstm: Whether to use LSTM in the model
        :param max_len: Maximum length for the model
        :param protein_dropout_rate: Dropout rate for protein features
        :param attention: Type of attention mechanism
        :param positional_encoding_type: Type of positional encoding
        :param pre_norm: Whether to use pre-normalization
        :param progressive_dropout: Whether to use progressive dropout
        :param initial_dropout_rate: Initial dropout rate for progressive dropout
        :param final_dropout_rate: Final dropout rate for progressive dropout
        :param output_dim: Output dimension for the model
        :param num_layers: Number of transformer layers
        :return: Loaded model
        """
        # Default output_dim to num_classes if not specified
        if output_dim is None:
            output_dim = num_classes
        
        # Read in the model state dict
        state_dict = torch.load(model_path, map_location=torch.device(self.device))

        logger.info(f'Reading model with input_dim={input_dim}, num_classes={num_classes}, output_dim={output_dim}, num_heads={num_heads}, hidden_dim={hidden_dim}, lstm_hidden_dim={lstm_hidden_dim}, dropout={dropout}, use_lstm={use_lstm}, max_len={max_len}, protein_dropout_rate={protein_dropout_rate}, attention={attention}, positional_encoding_type={positional_encoding_type}, pre_norm={pre_norm}, num_layers={num_layers}')

        # Select the positional encoding function based on the type
        positional_encoding_func = model_onehot.fourier_positional_encoding if positional_encoding_type == 'fourier' else model_onehot.sinusoidal_positional_encoding
        
        # Create the appropriate model based on attention type
        if attention == 'circular':
            model = model_onehot.TransformerClassifierCircularRelativeAttention(
                input_dim=input_dim, 
                num_classes=num_classes, 
                num_heads=num_heads, 
                hidden_dim=hidden_dim, 
                lstm_hidden_dim=lstm_hidden_dim if use_lstm else None,
                dropout=dropout, 
                max_len=max_len,
                use_lstm=use_lstm, 
                positional_encoding=positional_encoding_func,
                protein_dropout_rate=protein_dropout_rate,
                pre_norm=pre_norm,
                progressive_dropout=progressive_dropout,
                initial_dropout_rate=initial_dropout_rate,
                final_dropout_rate=final_dropout_rate,
                output_dim=output_dim
            )
        elif attention == 'relative':
            model = model_onehot.TransformerClassifierRelativeAttention(
                input_dim=input_dim,
                num_classes=num_classes,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                lstm_hidden_dim=lstm_hidden_dim if use_lstm else None,
                dropout=dropout,
                max_len=max_len,
                use_lstm=use_lstm,
                positional_encoding=positional_encoding_func,
                protein_dropout_rate=protein_dropout_rate,
                output_dim=output_dim
            )
        elif attention == 'absolute':
            model = model_onehot.TransformerClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                lstm_hidden_dim=lstm_hidden_dim if use_lstm else None,
                dropout=dropout,
                num_layers=num_layers,  # Use the passed parameter instead of hardcoded value
                use_lstm=use_lstm,
                positional_encoding=positional_encoding_func,
                protein_dropout_rate=protein_dropout_rate,
                output_dim=output_dim
            )
        else:
            raise ValueError(f"Invalid attention type: {attention}")
        
        # Initialize protein feature dropout with dummy forward pass
        if hasattr(model, 'protein_feature_dropout'):
            logger.info("Initializing protein_feature_dropout buffers with dummy forward pass")
            batch_size, seq_len = 2, 10  # Minimal batch for initialization
            feature_dim = model.gene_feature_dim + (hidden_dim - model.gene_feature_dim)
            dummy_input = torch.zeros((batch_size, seq_len, feature_dim), device=self.device)
            dummy_idx = [torch.tensor([0, 1]) for _ in range(batch_size)]
            model.protein_feature_dropout(dummy_input, dummy_idx)
        
        # Load the state dictionary with strict=False to handle missing/unexpected keys
        model.load_state_dict(state_dict, strict=False)
        
        return model

    def read_models_from_directory(self, directory_path, input_dim=1280, num_classes=9, num_heads=8, hidden_dim=512, lstm_hidden_dim=512, dropout=0.1, use_lstm=True, max_len=1500, protein_dropout_rate=0.6, attention='circular', positional_encoding_type='sinusoidal', pre_norm=False, progressive_dropout=True, initial_dropout_rate=1.0, final_dropout_rate=0.4, output_dim=None, num_layers=2):
        """
        Read and load all models from the given directory.

        :param directory_path: Path to the directory containing model files
        :param input_dim: Input dimension for the model
        :param num_classes: Number of classes for the model
        :param num_heads: Number of attention heads for the model
        :param hidden_dim: Hidden dimension for the model
        :param lstm_hidden_dim: LSTM hidden dimension for the model
        :param dropout: Dropout rate for the model
        :param use_lstm: Whether to use LSTM in the model
        :param max_len: Maximum length for the model
        :param protein_dropout_rate: Dropout rate for protein features
        :param attention: Type of attention mechanism
        :param positional_encoding_type: Type of positional encoding
        :param pre_norm: Whether to use pre-normalization
        :param progressive_dropout: Whether to use progressive dropout
        :param initial_dropout_rate: Initial dropout rate for progressive dropout
        :param final_dropout_rate: Final dropout rate for progressive dropout
        :param output_dim: Output dimension for the model
        :param num_layers: Number of transformer layers
        """
        print('Reading models from directory: ', directory_path)
        
        # Set default output_dim to num_classes if not specified
        if output_dim is None:
            output_dim = num_classes
        
        # Log parameters for debugging
        logger.info(f"Loading models with parameters: input_dim={input_dim}, num_classes={num_classes}, output_dim={output_dim}, " 
                    f"num_heads={num_heads}, hidden_dim={hidden_dim}, lstm_hidden_dim={lstm_hidden_dim}, "
                    f"dropout={dropout}, use_lstm={use_lstm}, max_len={max_len}, "
                    f"protein_dropout_rate={protein_dropout_rate}, attention={attention}, "
                    f"positional_encoding_type={positional_encoding_type}, pre_norm={pre_norm}")

        for filename in os.listdir(directory_path):
            if filename.endswith(".model"):  # Assuming model files have .model extension
                model_path = os.path.join(directory_path, filename)
                model = self.read_model(
                    model_path, input_dim, num_classes, num_heads, hidden_dim, 
                    lstm_hidden_dim, dropout, use_lstm, max_len, protein_dropout_rate,
                    attention, positional_encoding_type, pre_norm, progressive_dropout,
                    initial_dropout_rate, final_dropout_rate, output_dim, num_layers
                )
                self.models.append(model)


    def compute_confidence(self, scores, confidence_dict, categories):
        """
        Compute confidence of predictions in a highly efficient vectorized manner.
        Processes all genomes simultaneously for each category to minimize redundant operations.
        
        :param scores: List of score arrays for the genomes
        :param confidence_dict: Dictionary containing confidence information
        :param categories: Dictionary of categories
        :return: Tuple of predictions and confidence scores lists
        """
        # Convert to list if a single score array was provided
        if isinstance(scores, np.ndarray) and scores.ndim > 1:
            scores = [scores]
        
        num_genomes = len(scores)
        logger.info(f'Processing confidence for {num_genomes} genomes')
        
        # Create the category mapping once
        categories_map = dict(zip(range(len(confidence_dict.keys())), confidence_dict.keys()))
        
        # Calculate predictions for all genomes first
        all_predictions = []
        all_confidence = []
        
        # Pre-allocate arrays for results
        for i, genome_scores in enumerate(scores):
            # Get predictions for this genome
            predictions = np.argmax(genome_scores, axis=1)
            all_predictions.append(np.zeros_like(predictions, dtype=float))
            all_confidence.append(np.zeros(len(predictions), dtype=float))
        
        # Process each category once for all genomes
        for category in range(9):  # Assuming 9 categories (0-8)
            category_key = categories_map.get(category)
            if category_key not in confidence_dict:
                logger.warning(f"Category {category} (key={category_key}) not found in confidence dict")
                continue
            
            # Get the model for this category
            category_model = confidence_dict.get(category_key)
            
            # Skip if missing key components
            required_keys = ["kde_TP", "kde_FP", "num_TP", "num_FP"]
            if not all(key in category_model for key in required_keys):
                logger.warning(f"Missing required keys in confidence model for category {category}")
                continue
            
            # Process this category for all genomes at once
            for g, genome_scores in enumerate(scores):
                # Find positions where this category is predicted
                predictions = np.argmax(genome_scores, axis=1)
                category_mask = (predictions == category)
                
                # Skip if no predictions for this category in this genome
                if not np.any(category_mask):
                    continue
                    
                # Get relevant scores for this category
                category_scores = genome_scores[category_mask, category].reshape(-1, 1)
                
                try:
                    # Calculate confidence values for this category
                    e_TP = np.exp(category_model["kde_TP"].score_samples(category_scores))
                    e_FP = np.exp(category_model["kde_FP"].score_samples(category_scores))
                    
                    num_TP = category_model["num_TP"]
                    num_FP = category_model["num_FP"]
                    
                    # Calculate confidence scores
                    conf = (e_TP * num_TP) / (e_TP * num_TP + e_FP * num_FP + 1e-8)  # Add small epsilon to avoid division by zero
                    
                    # Store results
                    all_predictions[g][category_mask] = category
                    all_confidence[g][category_mask] = conf
                except Exception as e:
                    logger.error(f"Error calculating confidence for category {category}: {str(e)}")
                    # Set defaults
                    all_predictions[g][category_mask] = category
                    all_confidence[g][category_mask] = 0.5  # Conservative default
        
        logger.info(f"Completed confidence calculation for {num_genomes} genomes")
        return all_predictions, all_confidence

    def compute_confidence_isotonic(self, scores, calibration_models, phrog_integer):
        """
        Compute confidence of predictions using isotonic regression calibration models.
        
        :param scores: List of score arrays for the genomes
        :param calibration_models: Dictionary or list of dictionaries containing isotonic regression models for each class
        :param phrog_integer: Dictionary mapping phrog annotations to integer categories
        :return: Tuple of predictions, confidence scores lists, and dictionaries of raw and calibrated scores
        """
        # Convert to list if a single score array was provided
        if isinstance(scores, np.ndarray) and scores.ndim > 1:
            scores = [scores]
        
        num_genomes = len(scores)
        logger.info(f'Processing isotonic calibration for {num_genomes} genomes')
        
        # Check if we have per-model calibrations (list of model calibration dictionaries)
        use_ensemble = isinstance(calibration_models, list)
        num_models = len(calibration_models) if use_ensemble else 1
        
        if use_ensemble:
            logger.info(f"Using ensemble of {num_models} calibration models")
        else:
            logger.info("Using single calibration model")
            # Wrap in a list for consistent processing
            calibration_models = [calibration_models]
        
        # Create the category mapping from calibration_models keys
        # Use the first model's keys if we have an ensemble
        categories_map = {i: cat for i, cat in enumerate(calibration_models[0].keys())}
        
        # Pre-allocate arrays for results
        predictions = []
        confidence_scores = []
        raw_scores_dict = {}
        calibrated_scores_dict = {}
        
        # Process each genome
        for i, genome_scores in enumerate(scores):
            # Get the raw predicted class (highest score index)
            pred_indices = np.argmax(genome_scores, axis=1)
            genome_preds = []
            genome_confidences = []
            
            # For each gene in this genome
            for j, pred_idx in enumerate(pred_indices):
                # Get the raw score for the predicted class
                if pred_idx < len(genome_scores[j]):
                    raw_score = genome_scores[j][pred_idx]
                    
                    # Get the category name from the map
                    if pred_idx in categories_map:
                        cat_name = categories_map[pred_idx]
                        
                        # Apply calibration across all models and average results
                        calibrated_scores = []
                        
                        for model_idx in range(num_models):
                            if cat_name in calibration_models[model_idx]:
                                calibration_model = calibration_models[model_idx][cat_name]
                                # Reshape for sklearn models which expect 2D input
                                score_2d = np.array([[raw_score]])
                                try:
                                    cal_score = calibration_model.predict(score_2d)[0]
                                    calibrated_scores.append(cal_score)
                                except Exception as e:
                                    logger.error(f"Error in calibration model {model_idx} for category {cat_name}: {e}")
                            else:
                                logger.warning(f"No calibration model for category {cat_name} in model {model_idx}")
                        
                        # Average calibrated scores if we got any valid ones
                        if calibrated_scores:
                            calibrated_score = sum(calibrated_scores) / len(calibrated_scores)
                        else:
                            # Use raw score as fallback
                            calibrated_score = raw_score
                            
                        # Store predictions and confidence
                        genome_preds.append(pred_idx)
                        genome_confidences.append(calibrated_score)
                        
                        # Store raw and calibrated scores for return
                        if i not in raw_scores_dict:
                            raw_scores_dict[i] = {}
                            calibrated_scores_dict[i] = {}
                        raw_scores_dict[i][j] = raw_score
                        calibrated_scores_dict[i][j] = calibrated_score
                    else:
                        logger.warning(f"Category index {pred_idx} not in categories_map")
                        genome_preds.append(pred_idx)
                        genome_confidences.append(raw_score)
                else:
                    logger.warning(f"Prediction index {pred_idx} out of bounds for genome {i}, gene {j}")
                    # Use a safe fallback
                    genome_preds.append(0)
                    genome_confidences.append(0.0)
            
            predictions.append(np.array(genome_preds))
            confidence_scores.append(np.array(genome_confidences))
        
        logger.info(f"Completed confidence calculation for {num_genomes} genomes")
        return predictions, confidence_scores, raw_scores_dict, calibrated_scores_dict


########
# Confidence scoring functions
########
def count_critical_points(arr):
    """
    Count the number of critical points in an array.

    :param arr: Input array
    :return: Number of critical points
    """
    return np.sum(np.diff(np.sign(np.diff(arr))) != 0)
 
def build_confidence_dict(label, prediction, scores, bandwidth, categories):
    """
    Build a dictionary containing confidence information for each category.

    :param label: Array of true labels
    :param prediction: Array of predicted labels
    :param scores: Array of scores
    :param bandwidth: List of bandwidth values for kernel density estimation
    :param categories: Dictionary of categories
    :return: Dictionary containing confidence information
    """
    # range over values to compute kernel density over
    vals = np.arange(1.5, 10, 0.001)

    # save a dictionary which contains all the information required to compute confidence scores
    confidence_dict = dict()

    # loop through the categories
    print("Computing kernel density for each category...")
    for cat in range(0, 9):
        print(f"Processing category {cat}")

        # fetch the true labels of the predictions of this category
        this_labels = label[prediction == cat]

        # fetch the scores associated with these predictions
        this_scores = scores[prediction == cat]

        # separate false positives and true positives
        TP_scores = this_scores[this_labels == cat]
        FP_scores = this_scores[this_labels != cat]

        print(f"Category {cat}: TP_scores shape: {TP_scores.shape}, FP_scores shape: {FP_scores.shape}")

        if TP_scores.shape[0] == 0 or FP_scores.shape[0] == 0:
            logger.warning(f"Skipping category {cat} due to insufficient data (TP: {TP_scores.shape[0]}, FP: {FP_scores.shape[0]}).")
            continue

        # loop through potential bandwidths
        for b in bandwidth:
            try:
                # compute the kernel density
                kde_TP = KernelDensity(kernel="gaussian", bandwidth=b)
                kde_TP.fit(TP_scores[:, cat].reshape(-1, 1))
                e_TP = np.exp(kde_TP.score_samples(vals.reshape(-1, 1)))

                kde_FP = KernelDensity(kernel="gaussian", bandwidth=b)
                kde_FP.fit(FP_scores[:, cat].reshape(-1, 1))
                e_FP = np.exp(kde_FP.score_samples(vals.reshape(-1, 1)))

                conf_kde = (e_TP * len(TP_scores)) / (
                    e_TP * len(TP_scores) + e_FP * len(FP_scores) + 1e-8  # Add epsilon to avoid division by zero
                )

                if np.isnan(conf_kde).any():
                    logger.warning(f"NaN values encountered in confidence KDE for category {cat} with bandwidth {b}. Skipping.")
                    continue

                if count_critical_points(conf_kde) <= 1:
                    break
            except Exception as e:
                logger.error(f"Error during KDE computation for category {cat} with bandwidth {b}: {e}")
                logger.debug(traceback.format_exc())
                continue

        # save the best estimators
        confidence_dict[categories.get(cat)] = {
            "kde_TP": kde_TP,
            "kde_FP": kde_FP,
            "num_TP": len(TP_scores),
            "num_FP": len(FP_scores),
            "bandwidth": b,
        }

    return confidence_dict

def calculate_confidence(max_prob, prediction, confidence_dict):
    """
    Calculate confidence scores for predictions.

    :param max_prob: Maximum probability for each prediction
    :param prediction: Predicted category for each sample
    :param confidence_dict: Dictionary containing confidence information
    :return: List of confidence scores
    """
    logger.debug(f"Starting calculate_confidence with max_prob={max_prob}, prediction={prediction}")
    logger.debug(f"Type of max_prob: {type(max_prob)}, Type of prediction: {type(prediction)}")

    # Ensure max_prob and prediction are iterable
    if not isinstance(max_prob, (list, np.ndarray)):
        logger.warning(f"max_prob is not iterable. Wrapping it in a list: {max_prob}")
        max_prob = [max_prob]
    if not isinstance(prediction, (list, np.ndarray)):
        logger.warning(f"prediction is not iterable. Wrapping it in a list: {prediction}")
        prediction = [prediction]

    confidence_scores = []
    for prob, pred in zip(max_prob, prediction):
        try:
            logger.debug(f"Processing prob={prob}, pred={pred}")
            if pred in confidence_dict:
                kde_TP = confidence_dict[pred].get("kde_TP")
                kde_FP = confidence_dict[pred].get("kde_FP")
                num_TP = confidence_dict[pred].get("num_TP", 0)
                num_FP = confidence_dict[pred].get("num_FP", 0)

                if kde_TP is None or kde_FP is None or num_TP == 0 or num_FP == 0:
                    logger.warning(f"Missing or invalid KDE data for category {pred}. Assigning default confidence of 0.")
                    confidence_scores.append(0)
                    continue

                e_TP = np.exp(kde_TP.score_samples([[prob]]))
                e_FP = np.exp(kde_FP.score_samples([[prob]]))
                conf_score = (e_TP * num_TP) / (e_TP * num_TP + e_FP * num_FP + 1e-8)  # Add epsilon to avoid division by zero

                if np.isnan(conf_score).any():
                    logger.warning(f"NaN confidence score encountered for category {pred}. Assigning default confidence of 0.")
                    conf_score = 0

                confidence_scores.append(conf_score)
            else:
                logger.warning(f"Category {pred} not found in confidence dictionary. Assigning default confidence of 0.")
                confidence_scores.append(0)
        except Exception as e:
            logger.error(f"Error calculating confidence for prob={prob}, pred={pred}: {e}")
            logger.debug(traceback.format_exc())
            confidence_scores.append(0)  # Default confidence if an error occurs

    logger.debug(f"Finished calculate_confidence. Confidence scores: {confidence_scores}")
    return confidence_scores
