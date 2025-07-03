"""Endpoints for getting version information."""
from typing import Any
from fastapi import APIRouter, Body
from ..schemas.base import VersionResponse, FrontalCropSubmitRequest, FrontalCropSubmitResponse
from ..version import __version__
import base64

base_router = APIRouter()


@base_router.get("/version", response_model=VersionResponse)
async def version() -> Any:
    """Provide version information about the web service.

    \f
    Returns:
        VersionResponse: A json response containing the version number.
    """
    return VersionResponse(version=__version__)


@base_router.post("/frontal/crop/submit", response_model=FrontalCropSubmitResponse)
async def frontal_crop_submit(
    payload: FrontalCropSubmitRequest = Body(...)
) -> FrontalCropSubmitResponse:
    """Process a portrait image, landmarks, and segmentation map, returning SVG and mask contours."""
    # Here you would process the image, landmarks, and segmentation map.
    # For demonstration, we return dummy data.
    dummy_svg = base64.b64encode(b'<svg></svg>').decode()
    dummy_mask_contours = {1: [[0, 0], [1, 1], [2, 2]]}
    return FrontalCropSubmitResponse(svg=dummy_svg, mask_contours=dummy_mask_contours)
