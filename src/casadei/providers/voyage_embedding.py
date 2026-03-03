"""Voyage AI Embedding Provider."""

from __future__ import annotations

import os
from typing import Any

from casadei.media import MediaBundle, TextMedia, EmbeddingMedia
from casadei.models.base import ModelCapability, TextConstraint
from casadei.models.embedding import EmbeddingModel, EmbeddingConstraint

import voyageai


class VoyageEmbeddingProvider(EmbeddingModel):
    """Provider for voyage-4 text embedding API.
    
    Accepts multiple TextMedia inputs and returns one EmbeddingMedia
    containing lists of floats corresponding to the text embeddings.
    """

    DEFAULT_PARAMS = {
        "model": "voyage-4",
        "input_type": "document",
    }

    def __init__(self, api_key: str | None = None) -> None:
        self.capability = ModelCapability(
            inputs=[
                TextConstraint(required=True, max_count=100) # Accept up to 100 text documents 
            ],
            outputs=[
                EmbeddingConstraint(required=True, max_count=1)
            ],
        )
        super().__init__()
        
        # Will use environment variable VOYAGE_API_KEY if not provided
        self.client = voyageai.Client(api_key=api_key)

    def load_model(self) -> None:
        """No local model weights to load."""
        pass

    def unload_model(self) -> None:
        """No local model weights to unload."""
        pass

    def run(self, inputs: MediaBundle, **kwargs) -> MediaBundle:
        """Run voyage embedding inference.
        
        Extracts all TextMedia items from the input bundle.
        """
        # Validate inputs
        errors = self.capability.validate_inputs(inputs)
        if errors:
            raise ValueError(f"Input validation failed: {errors}")

        # Gather all text data
        texts = []
        for v in inputs.items.values():
            if isinstance(v, TextMedia):
                texts.append(v.text)

        if not texts:
            raise ValueError("No text provided to embed.")

        params = self.get_all_params()
        
        # Override with any explicit kwargs
        model_kwargs = {**params, **kwargs}
        
        # Extract specific voyage args
        model_name = model_kwargs.pop("model")
        input_type = model_kwargs.pop("input_type")
        
        # Call voyage client
        result = self.client.embed(
            texts=texts,
            model=model_name,
            input_type=input_type,
            **model_kwargs
        )
        
        # Results is an object with an embeddings list: [[...], [...]]
        # result.embeddings (list of lists of floats)
        
        out_media = EmbeddingMedia(embeddings=result.embeddings)
        
        return MediaBundle(items={"embeddings": out_media})
