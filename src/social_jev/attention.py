"""Packed visibility masks and logical token positions."""

from typing import TYPE_CHECKING

import torch
from torch.nn.attention.flex_attention import create_block_mask

if TYPE_CHECKING:
    from .data import Batch


def block_mask(ranges):
    """Build a FlexAttention block mask from allowed token ranges."""
    batch, length = ranges.shape[:2]

    def allowed(b, _h, q, k):
        """Return whether a query token may attend to a key token."""
        valid = (q < length) & (k < length)
        bounds = ranges[b, q.clamp(max=length - 1)]
        return valid & (
            ((k >= bounds[0, 0]) & (k < bounds[0, 1]))
            | ((k >= bounds[1, 0]) & (k < bounds[1, 1]))
            | ((k >= bounds[2, 0]) & (k < bounds[2, 1]))
        )

    return create_block_mask(allowed, batch, None, length, length, device=ranges.device)


def message_spans(boundaries, roles, answer_length):
    """Expose the appended answer prefix without changing stored message roles."""
    start = 0
    for index, (boundary, role) in enumerate(zip(boundaries, roles, strict=True)):
        end = boundary + 1
        final_option = role == "option" and (
            index + 1 == len(roles) or roles[index + 1] != "option"
        )
        if answer_length and final_option:
            yield start, end - answer_length, role
            yield end - answer_length, end, "answer"
        else:
            yield start, end, role
        start = end


class Positions:
    """Compute logical token positions for isolated queries and options.

    Args:
        batch: Padded tokens and their message boundaries.
        answer_length: Number of tokens in each appended answer prefix.

    Examples:
        >>> positions = Positions(batch, answer_length=2)  # doctest: +SKIP
        >>> positions.value.shape == batch.input_ids.shape  # doctest: +SKIP
        True
    """

    def __init__(  # noqa: C901 -- One branch per message role.
        self, batch: "Batch", *, answer_length: int = 0
    ):
        """Build logical positions for all tokens in the batch."""
        positions = []
        for boundaries, roles in zip(batch.message_boundaries, batch.roles, strict=True):
            sequence = []
            state_length = query_length = longest_example = longest_option = 0
            for start, end, role in message_spans(boundaries, roles, answer_length):
                length = end - start
                if role == "state":
                    state_length = length
                    query_length = longest_example = longest_option = 0
                    offset = 0
                elif role == "query":
                    query_length = length
                    longest_example = longest_option = 0
                    offset = state_length
                elif role == "example":
                    offset = state_length + query_length
                    longest_example = max(longest_example, length)
                elif role == "answer":
                    offset = state_length + query_length + longest_example + longest_option
                else:  # option; Batch validates supported roles.
                    longest_option = max(longest_option, length)
                    offset = state_length + query_length + longest_example
                sequence.extend(range(offset, offset + length))
            padding = batch.input_ids.shape[1] - len(sequence)
            positions.append(sequence + [0] * padding)
        self.value = torch.tensor(positions, dtype=torch.long, device=batch.input_ids.device)


class AttentionMask:
    """Compute causal visibility within documents, queries, and candidate answers.

    Args:
        batch: Padded tokens and their message boundaries.
        answer_length: Number of tokens in each appended answer prefix.

    Examples:
        >>> mask = AttentionMask(batch, answer_length=2)  # doctest: +SKIP
        >>> block_mask = mask.value  # doctest: +SKIP
    """

    def __init__(  # noqa: C901 -- One branch per message role.
        self, batch: "Batch", *, answer_length: int = 0
    ):
        """Build the visibility mask for all tokens in the batch."""
        batch_size, length = batch.input_ids.shape
        tokens = torch.arange(length)
        ranges = torch.zeros((batch_size, length, 3, 2), dtype=torch.int32)
        ranges[:, :, 0, 0] = tokens
        ranges[:, :, 0, 1] = tokens + 1
        for index, (boundaries, roles) in enumerate(
            zip(batch.message_boundaries, batch.roles, strict=True)
        ):
            state = query = (0, 0)
            context_end = 0
            for start, end, role in message_spans(boundaries, roles, answer_length):
                ranges[index, start:end, 0, 0] = start
                ranges[index, start:end, 0, 1] = torch.arange(start + 1, end + 1)
                if role == "state":
                    state = (start, end)
                    query = (0, 0)
                    context_end = 0
                elif role == "query":
                    query = (start, end)
                    context_end = end
                    ranges[index, start:end, 1] = torch.tensor(state)
                elif role == "example":
                    ranges[index, start:end, 1] = torch.tensor(state)
                    ranges[index, start:end, 2] = torch.tensor(query)
                    context_end = end
                elif role == "answer":
                    ranges[index, start:end, 1] = torch.tensor(state)
                    ranges[index, start:end, 2] = torch.tensor([query[0], start])
                else:
                    ranges[index, start:end, 1] = torch.tensor(state)
                    ranges[index, start:end, 2] = torch.tensor([query[0], context_end])
        self.value = block_mask(ranges.to(batch.input_ids.device))
