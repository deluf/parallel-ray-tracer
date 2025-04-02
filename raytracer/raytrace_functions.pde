//RICONTROLLARE L'ANGOLO DI RIFLESSIONE E IL BLINN PHONG SHADING

//constant used to remove 'acne' effect
float epsilon = 1e-3;

float[] rayTrace(Vector origin, Vector dir, int limit){
  float[] col = new float[3];
  
  if(limit == LIMIT)
    return col;
  
  int index = -1;
  float dist = Float.MAX_VALUE;
  float t = -1;
  int type = 0;
  //check nearest sphere
  for(int i = 0; i < spheres.length; i++){
    float t_tmp = hitCollision(origin, dir, spheres[i]);
    if(t_tmp > epsilon){
      Vector intersection = Vector.add(origin, Vector.mult(dir, t_tmp));
      float d = Vector.dist(origin, intersection);
      if(d < dist){
        index = i;
        dist = d;
        t = t_tmp;
        type = 0;
      }
    }
  }
  //check nearest triangle
  for(int i = 0; i < triangles.length; i++){
    float t_tmp = hitCollision(origin, dir, triangles[i]);
    if(t_tmp > epsilon){
      Vector intersection = Vector.add(origin, Vector.mult(dir, t_tmp));
      float d = Vector.dist(origin, intersection);
      if(d < dist){
        index = i;
        dist = d;
        t = t_tmp;
        type = 1;
      }
    }
  }
  
  //INDEX | T | TYPE
  //TYPE = 0 : SPHERE
  //TYPE = 1 : TRIANGLE
   
  if(index < 0){
    if(!skybox.exists){
      col[0] = ambLight.kl[0];
      col[1] = ambLight.kl[1];
      col[2] = ambLight.kl[2];
    } else {
      float[] sky_col = skybox.getColor(dir);
      col[0] = sky_col[0];
      col[1] = sky_col[1];
      col[2] = sky_col[2];
    }
  } else {
    Vector intersection = Vector.add(origin, Vector.mult(dir, t));
    //if intersection is sphere
    float[] ks = new float[3];
    float[] kd = new float[3];
    float[] kr = {0.5, 0.5, 0.5};
    Vector n = new Vector(0, 0, 0);
    if(type == 0){
      ks = spheres[index].ks;
      kd = spheres[index].kd;
      n = Vector.sub(intersection, spheres[index].pos).normalize();
    }
    if(type == 1){
      ks = triangles[index].ks;
      kd = triangles[index].kd;
      kr = triangles[index].kr;
      n = triangles[index].norm[0];
    }
    //apply ambient light
    col[0] = kd[0]*ambLight.kl[0];
    col[1] = kd[1]*ambLight.kl[1];
    col[2] = kd[2]*ambLight.kl[2];
    //apply directional lights
    dir.mult(-1);
    for(int i = 0; i < dirLights.length; i++){
      float n_dot_l = Vector.dot(dirLights[i].dir, n);
      float[] col_ray = lambertBlinnShading(ks, kd, n, dirLights[i].dir, dir, n_dot_l);
      int V = V(intersection, dirLights[i].dir);
      col[0] += V*dirLights[i].kl[0]*col_ray[0];
      col[1] += V*dirLights[i].kl[0]*col_ray[1];
      col[2] += V*dirLights[i].kl[0]*col_ray[2];
    }
    //apply point lights
    for(int i = 0; i < pointLights.length; i++){
      Vector l = Vector.sub(pointLights[i].pos, intersection);
      float mag = l.mag();
      l.div(mag);
      mag *= mag;
      float n_dot_l = Vector.dot(n, l);
      float[] col_ray = lambertBlinnShading(ks, kd, n, l, dir, n_dot_l);
      int V = V(intersection, l, pointLights[i].pos);
      col[0] += V*pointLights[i].kl[0]/mag*col_ray[0];
      col[1] += V*pointLights[i].kl[0]/mag*col_ray[1];
      col[2] += V*pointLights[i].kl[0]/mag*col_ray[2];
    }
    
    //real raytracing EXTREMELY HEAVY
    dir.mult(-1);
    Vector r = Vector.add(dir, Vector.mult(n, 2*abs(Vector.dot(dir, n)))).normalize();
    float[]  col_ray = rayTrace(intersection, r, limit+1);
    col[0] += kr[0]*col_ray[0];
    col[1] += kr[1]*col_ray[1];
    col[2] += kr[2]*col_ray[2];
    
  }
  col[0] = constrain(col[0], 0, 1);
  col[1] = constrain(col[1], 0, 1);
  col[2] = constrain(col[2], 0, 1);
    
  return col;
}

//FOR DIRECTIONAL LIGHTS
int V(Vector origin, Vector dir){
  //check nearest sphere
  for(int i = 0; i < spheres.length; i++){
    float t = hitCollision(origin, dir, spheres[i]);
    if(t > epsilon)
      return 0;
  }
  //check nearest triangle
  for(int i = 0; i < triangles.length; i++){
    float t = hitCollision(origin, dir, triangles[i]);
    if(t > epsilon)
      return 0;
  }
  return 1;
}

//FOR POINT LIGHTS
int V(Vector origin, Vector dir, Vector light){
  float lightDist = Vector.dot(Vector.sub(origin, light), Vector.sub(origin, light));
  //check nearest sphere
  for(int i = 0; i < spheres.length; i++){
    float t = hitCollision(origin, dir, spheres[i]);
    if(t > epsilon){
      Vector intersection = Vector.add(origin, Vector.mult(dir, t));
      if(lightDist > Vector.dot(Vector.sub(origin, intersection), Vector.sub(origin, intersection)) )
        return 0;
    }
  }
  //check nearest triangle
  for(int i = 0; i < triangles.length; i++){
    float t = hitCollision(origin, dir, triangles[i]);
    if(t > epsilon){
      Vector intersection = Vector.add(origin, Vector.mult(dir, t));
      if(lightDist > Vector.dot(Vector.sub(origin, intersection), Vector.sub(origin, intersection)) )
        return 0;
    }
  }
  return 1;
}

float hitCollision(Vector origin, Vector dir, Sphere sphere){
  float a = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z;
  float b = -2*( dir.x*(sphere.pos.x-origin.x) + dir.y*(sphere.pos.y-origin.y) + dir.z*(sphere.pos.z-origin.z) );
  float c = (sphere.pos.x-origin.x)*(sphere.pos.x-origin.x) + (sphere.pos.y-origin.y)*(sphere.pos.y-origin.y) + (sphere.pos.z-origin.z)*(sphere.pos.z-origin.z) - sphere.r*sphere.r;    
  float delta = b*b-4*a*c;
  if(delta < 0){
    return -1;
  } 
  float t1 = (-b + sqrt(delta))/2*a;
  float t2 = (-b - sqrt(delta))/2*a;
  float min = min(t1, t2);
  float max = max(t1, t2);
  if(min <= 0)
    return max;
  else
    return min;
}

float hitCollision(Vector origin, Vector dir, Triangle triangle){
  Vector e1 = Vector.sub(triangle.coords[1], triangle.coords[0]);
  Vector e2 = Vector.sub(triangle.coords[2], triangle.coords[0]);
  Vector n = e1.cross(e2);
  float det = -Vector.dot(dir, n);
  float invdet = 1.0/det;
  Vector ao = Vector.sub(origin, triangle.coords[0]);
  Vector dao = ao.cross(dir);
  float u = Vector.dot(e2, dao)*invdet;
  float v = -Vector.dot(e1, dao)*invdet;
  float t = Vector.dot(ao, n)*invdet;
  if(det > 0 && t > 0 && u > 0 && v > 0 && (u+v) < 1)
    return t;  
  
  e2 = Vector.sub(triangle.coords[1], triangle.coords[0]);
  e1 = Vector.sub(triangle.coords[2], triangle.coords[0]);
  n = e1.cross(e2);
  det = -Vector.dot(dir, n);
  invdet = 1.0/det;
  ao = Vector.sub(origin, triangle.coords[0]);
  dao = ao.cross(dir);
  u = Vector.dot(e2, dao)*invdet;
  v = -Vector.dot(e1, dao)*invdet;
  t = Vector.dot(ao, n)*invdet;
  if(det > 0 && t > 0 && u > 0 && v > 0 && (u+v) < 1)
    return t;  
  return -1;
}

float[] lambertBlinnShading(float[] ks, float[] kd, Vector n, Vector l, Vector v, float dot){
  Vector h = Vector.add(l, v);
  h.div(h.mag());
  
  float coeff = max(0, Vector.dot(n, h));
  
  float red = min((kd[0]*max(0, dot)+ks[0]*coeff), 1);
  float green = min((kd[1]*max(0, dot)+ks[1]*coeff), 1);
  float blue = min((kd[2]*max(0, dot)+ks[2]*coeff), 1);
  
  float[] out = {red, green, blue};
  return out;
}
