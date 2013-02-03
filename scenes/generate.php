<?php

Generator::main($argc, $argv);

class Generator {
   public static function main($argc, $argv) {
      $mode = false;
      $settings = self::get_settings($argc, $argv);
      extract($settings, EXTR_IF_EXISTS);

      if ($mode == 's') {
         $spheres = self::make_spheres($settings);
         self::print_spheres($spheres);
      } else if ($mode == 'l') {
         $lights = self::make_lights($settings);
         self::print_lights($lights);
      }

      exit(0);
   }
   private static function get_settings($argc, $argv) {
      $aspect_ratio = 16 / 9;
      $z_near = 10;
      $z_far = 200;
      $z_focal = 50;
      $hfov = 60 * M_PI / 180;
      $vfov = $hfov / $aspect_ratio;

      $num_spheres = 100;
      $min_spheres = 1;
      $max_spheres = 65535;
      $min_radius = 0.1;
      $max_radius = 2.5;
      $max_shine = 64;

      $num_lights = 1;
      $min_lights = 1;
      $max_lights = 8;

      $settings = array(
         'aspect_ratio' => $aspect_ratio,
         'z_near' => $z_near,
         'z_far' => $z_far,
         'z_focal' => $z_focal,
         'hfov' => $hfov,
         'zfov' => $vfov,

         'num_spheres' => $num_spheres,
         'min_radius' => $min_radius,
         'max_radius' => $max_radius,
         'max_shine' => $max_shine,

         'num_lights' => $num_lights
      );

      if ($argc > 1) {
         switch (strtolower($argv[1])) {
            case 's': case 'sphere': case 'spheres':
               $mode = 's';
               break;
            case 'l': case 'light': case 'spheres':
               $mode = 'l';
               break;
            default:
               $mode = 'u';
               break;
         }
         $settings['mode'] = $mode;

         if ($argc > 2) {
            $num = $argv[2];

            if ($mode == 's') {
               if ($num < $min_spheres || $num > $max_spheres) {
                  exit(1);
               }

               $settings['num_spheres'] = $num;
            } else if ($mode == 'l') {
               if ($num < $min_lights || $num > $max_lights) {
                  exit(1);
               }

               $settings['num_lights'] = $num;
            }
         }
      }

      return $settings;
   }

   private static function unit_rand() {
      return rand(0, 1000) / 1000;
   }
   private static function rand_in_range($range) {
      return self::unit_rand() * ($range[1] - $range[0]) + $range[0];
   }
   private static function rand_vec() {
      $unit_range = array(0, 1);
      return self::rand_vec_in_range(array(
         $unit_range,
         $unit_range,
         $unit_range
      ));
   }
   private static function rand_vec_in_range($ranges) {
      return array(
         self::rand_in_range($ranges[0]),
         self::rand_in_range($ranges[1]),
         self::rand_in_range($ranges[2])
      );
   }

   private static function print_vec($vec) {
      foreach ($vec as $val) {
         echo "$val ";
      }
      echo "\n";
   }

   private static function get_pos_in_view($z_near, $z_far, $hfov, $vfov) {
      $position = self::rand_vec();

      $z = $position[2] * ($z_far - $z_near) + $z_near;
      $max_x = tan($hfov / 2) * $z;
      $max_y = tan($vfov / 2) * $z;
      $x = $position[0] * (2 * $max_x) - $max_x;
      $y = $position[1] * (2 * $max_y) - $max_y;

      return array($x,$y, $z);
   }

   private static function make_light($settings) {
      $z_near = $z_far = $hfov = $vfov = false;
      extract($settings, EXTR_IF_EXISTS);

      return $light = array(
         'position' => self::get_pos_in_view($z_near, $z_far, $hfov, $vfov),
         'color' => self::rand_vec(),
      );
   }
   private static function make_lights($settings) {
      $num_lights = false;
      extract($settings, EXTR_IF_EXISTS);

      $lights = array();

      for ($ndx = 0; $ndx < $num_lights; ++$ndx) {
         $lights[] = self::make_light($settings);
      }

      return $lights;
   }
   private static function print_light($light) {
      self::print_vec($light['position']);
      self::print_vec($light['color']);
   }
   private static function print_lights($lights) {
      foreach ($lights as $light) {
         self::print_light($light);
         echo "\n";
      }
   }

   private static function make_sphere($settings) {
      $z_near = $z_far = $hfov = $vfov =
         $min_radius = $max_radius = $max_shine = false;
      extract($settings, EXTR_IF_EXISTS);

      return $sphere = array(
         'position' => self::get_pos_in_view($z_near, $z_far, $hfov, $vfov),
         'radius' => self::rand_in_range(array($min_radius, $max_radius)),
         'ambient' => self::rand_vec(),
         'diffuse' => self::rand_vec(),
         'specular' => self::rand_vec(),
         'shine' => self::unit_rand() * $max_shine
      );
   }
   private static function make_spheres($settings) {
      $num_spheres = false;
      extract($settings, EXTR_IF_EXISTS);

      $spheres = array();

      for ($ndx = 0; $ndx < $num_spheres; ++$ndx) {
         $spheres[] = self::make_sphere($settings);
      }

      return $spheres;
   }
   private static function print_sphere($sphere) {
      self::print_vec($sphere['position']);
      echo "{$sphere['radius']}\n";
      self::print_vec($sphere['ambient']);
      self::print_vec($sphere['diffuse']);
      self::print_vec($sphere['specular']);
      echo "{$sphere['shine']}\n";
   }
   private static function print_spheres($spheres) {
      foreach ($spheres as $sphere) {
         self::print_sphere($sphere);
         echo "\n";
      }
   }
}
