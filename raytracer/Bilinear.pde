float bilinear(float corner00, float corner10, float corner01, float corner11, float valX, float valY){
  float interpolated0 = lerp(corner00, corner10, valX);
  float interpolated1 = lerp(corner01, corner11, valX);
  return lerp(interpolated0, interpolated1, valY);
}
