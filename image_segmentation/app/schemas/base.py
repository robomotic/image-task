"""Define response model for the endpoint version."""
from pydantic import BaseModel, Field  # type: ignore


class VersionResponse(BaseModel):
    """Response for version endpoint."""
    version: str = Field(..., example="1.0.0")


class Landmark(BaseModel):
    x: float
    y: float


class FrontalCropSubmitRequest(BaseModel):
    image: str  # base64-encoded image
    landmarks: list[Landmark]
    segmentation_map: str  # base64-encoded segmentation map


class FrontalCropSubmitResponse(BaseModel):
    svg: str  # base64-encoded SVG
    mask_contours: dict[int, list]
