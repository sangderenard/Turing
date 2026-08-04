module kernel_fortran
  use, intrinsic :: iso_c_binding
  implicit none
contains

  subroutine columnar_multifluid_rgb_step(extent_16, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t514, t520, t526, t210, t208, t131, t139, t121, t123, t21, t396, t410, t424, t438, t452, t466) bind(C, name="columnar_multifluid_rgb_step")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_int), intent(in), value :: extent_16
    real(c_double), intent(in) :: t0(extent_16)
    real(c_double), intent(in) :: t1(extent_16)
    real(c_double), intent(in) :: t2(extent_16)
    real(c_double), intent(in) :: t3(extent_16)
    real(c_double), intent(in) :: t4(extent_16)
    real(c_double), intent(in) :: t5(extent_16)
    real(c_double), intent(in) :: t6(extent_16)
    real(c_double), intent(in) :: t7(extent_16)
    real(c_double), intent(in) :: t8(extent_16)
    real(c_double), intent(in) :: t9(extent_16)
    real(c_double), intent(in) :: t10(extent_16)
    real(c_double), intent(in) :: t11(extent_16)
    real(c_double), intent(in) :: t12(extent_16)
    real(c_double), intent(in) :: t13(extent_16)
    real(c_double), intent(in) :: t14(extent_16)
    real(c_double), intent(in) :: t15(extent_16)
    real(c_double), intent(in) :: t16(extent_16)
    real(c_double), intent(in) :: t17(extent_16)
    real(c_double), intent(in) :: t18(extent_16)
    real(c_double), intent(in) :: t19(extent_16)
    real(c_double), intent(in) :: t20(extent_16)
    real(c_double), intent(out) :: t514(extent_16)
    real(c_double), intent(out) :: t520(extent_16)
    real(c_double), intent(out) :: t526(extent_16)
    real(c_double), intent(out) :: t210(extent_16)
    real(c_double), intent(out) :: t208(extent_16)
    real(c_double), intent(out) :: t131(extent_16)
    real(c_double), intent(out) :: t139(extent_16)
    real(c_double), intent(out) :: t121(extent_16)
    real(c_double), intent(out) :: t123(extent_16)
    real(c_double), intent(out) :: t21(extent_16)
    real(c_double), intent(out) :: t396(extent_16)
    real(c_double), intent(out) :: t410(extent_16)
    real(c_double), intent(out) :: t424(extent_16)
    real(c_double), intent(out) :: t438(extent_16)
    real(c_double), intent(out) :: t452(extent_16)
    real(c_double), intent(out) :: t466(extent_16)
    real(c_double) :: t22(extent_16)
    real(c_double) :: t58(extent_16)
    real(c_double) :: t59(extent_16)
    real(c_double) :: t288(extent_16)
    real(c_double) :: t26(extent_16)
    real(c_double) :: t62(extent_16)
    real(c_double) :: t50(extent_16)
    real(c_double) :: t66(extent_16)
    real(c_double) :: t78(extent_16)
    real(c_double) :: t33(extent_16)
    real(c_double) :: t160(extent_16)
    real(c_double) :: t83
    real(c_double) :: t79(extent_16)
    real(c_double) :: t167(extent_16)
    real(c_double) :: t184(extent_16)
    real(c_double) :: t193(extent_16)
    real(c_double) :: t508(extent_16)
    real(c_double) :: t474(extent_16)
    real(c_double) :: t477(extent_16)
    real(c_double) :: t230(extent_16)
    real(c_double) :: t221(extent_16)

    ! block entry
    t21 = (t0 + t1)
    t22 = (t2 - t3)
    t58 = (t5 - t7)
    t59 = (t6 - t8)
    t288 = (t21 * 0.42_c_double)
    t26 = (((t4 * 2.0_c_double) - t2) - t3)
    t62 = ((t58 * t58) + (t59 * t59))
    t50 = (((t5 * 0.61_c_double) + (t6 * 0.83_c_double)) + (sin(((t5 * 0.37_c_double) - (t6 * 0.29_c_double))) * 0.72_c_double))
    t66 = sqrt((t62 + 0.12_c_double))
    t78 = (exp(((-t62) / 11.045000000000002_c_double)) / (t62 + 0.18_c_double))
    t33 = sqrt((((t22 * t22) + (t26 * t26)) + 1.0e-05_c_double))
    t160 = ((min(max(((((t2 * 0.42_c_double) + (t4 * 0.34_c_double)) + (t3 * 0.18_c_double)) + (t11 * 0.35_c_double)), 0.0_c_double), 1.0_c_double) * 0.3_c_double) + 0.32_c_double)
    t83 = (sum(t78) + 1.0e-06_c_double)
    t79 = (t78 * ((cos(t50) * (t22 / t33)) + (sin(t50) * (t26 / t33))))
    t121 = (t9 + ((((((t5 * 0.0_c_double) + (sum(((t58 / t66) * t79)) / t83)) * 1.85_c_double) + ((5.0_c_double - t7) * 0.1_c_double)) - (t9 * 0.72_c_double)) * t1))
    t123 = (t10 + ((((((t6 * 0.0_c_double) + (sum(((t59 / t66) * t79)) / t83)) * 1.85_c_double) + ((3.5_c_double - t8) * 0.1_c_double)) - (t10 * 0.72_c_double)) * t1))
    t131 = min(max((t7 + (t121 * t1)), 0.65_c_double), 9.35_c_double)
    t139 = min(max((t8 + (t123 * t1)), 0.65_c_double), 6.35_c_double)
    t167 = (((t5 - t131) * (t5 - t131)) + ((t6 - t139) * (t6 - t139)))
    t184 = (min(max((t160 - abs((t5 - t131))), 0.0_c_double), max((t160 - abs((t6 - t139))), 0.0_c_double)) / t160)
    t193 = exp(((-t167) / 3.6450000000000005_c_double))
    t508 = (t184 * t184)
    t396 = min(((t15 * exp((t1 * -0.05_c_double))) + (((t1 * 2.8_c_double) * exp(((-t167) / 0.3872_c_double))) * max(cos(t288), 0.0_c_double))), 1.0_c_double)
    t410 = min(((t16 * exp((t1 * -0.054_c_double))) + (((t1 * 2.8_c_double) * exp(((-t167) / 0.4608_c_double))) * max(cos((t288 - 1.0471975511965976_c_double)), 0.0_c_double))), 1.0_c_double)
    t424 = min(((t17 * exp((t1 * -0.058_c_double))) + (((t1 * 2.8_c_double) * exp(((-t167) / 0.5408000000000001_c_double))) * max(cos((t288 - 2.0943951023931953_c_double)), 0.0_c_double))), 1.0_c_double)
    t438 = min(((t18 * exp((t1 * -0.062_c_double))) + (((t1 * 2.8_c_double) * exp(((-t167) / 0.6272000000000001_c_double))) * max(cos((t288 - 3.141592653589793_c_double)), 0.0_c_double))), 1.0_c_double)
    t452 = min(((t19 * exp((t1 * -0.066_c_double))) + (((t1 * 2.8_c_double) * exp(((-t167) / 0.72_c_double))) * max(cos((t288 - 4.1887902047863905_c_double)), 0.0_c_double))), 1.0_c_double)
    t466 = min(((t20 * exp((t1 * -0.07_c_double))) + (((t1 * 2.8_c_double) * exp(((-t167) / 0.8192_c_double))) * max(cos((t288 - 5.235987755982989_c_double)), 0.0_c_double))), 1.0_c_double)
    t208 = (t13 + ((((((t193 * -0.42_c_double) - ((t184 * 0.22_c_double) * t184)) - t12) * 20.0_c_double) - (t13 * 8.0_c_double)) * t1))
    t210 = (t12 + (t208 * t1))
    t474 = max((((((t396 + t410) + t424) + t438) + t452) + t466), 1.0e-06_c_double)
    t477 = min(t474, 0.88_c_double)
    t230 = min(max(((-t210) / 0.42_c_double), 0.0_c_double), 1.0_c_double)
    t221 = min(max((((t14 + t210) - 0.5_c_double) / 5.0_c_double), 0.0_c_double), 1.0_c_double)
    t514 = ((((min(max(((((t221 * 27.0_c_double) + 186.0_c_double) - (t230 * 34.0_c_double)) + (t193 * 54.0_c_double)), 0.0_c_double), 255.0_c_double) * (1.0_c_double - t477)) + (((((t396 + t410) + t466) * 255.0_c_double) / t474) * t477)) * (1.0_c_double - t508)) + (t508 * 245.0_c_double))
    t526 = ((((min(max(((((t221 * 16.0_c_double) + 232.0_c_double) + (t230 * 15.0_c_double)) + (t193 * 20.0_c_double)), 0.0_c_double), 255.0_c_double) * (1.0_c_double - t477)) + (((((t438 + t452) + t466) * 255.0_c_double) / t474) * t477)) * (1.0_c_double - t508)) + (t508 * 255.0_c_double))
    t520 = ((((min(max((((((t221 * 18.0_c_double) + 220.0_c_double) - (t230 * 21.0_c_double)) + (t193 * 30.0_c_double)) + (min(abs(t208), 1.0_c_double) * 8.0_c_double)), 0.0_c_double), 255.0_c_double) * (1.0_c_double - t477)) + (((((t410 + t424) + t438) * 255.0_c_double) / t474) * t477)) * (1.0_c_double - t508)) + (t508 * 252.0_c_double))
    return
  end subroutine columnar_multifluid_rgb_step
  subroutine columnar_multifluid_rgb_step_control(extent_1, extent_16, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t514, t520, t526, t210, t208, t131, t139, t121, t123, t21, t396, t410, t424, t438, t452, t466) bind(C, name="columnar_multifluid_rgb_step_control")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_int), intent(in), value :: extent_1
    integer(c_int), intent(in), value :: extent_16
    real(c_double), intent(in) :: t0(extent_16)
    real(c_double), intent(in) :: t1(extent_16)
    real(c_double), intent(in) :: t2(extent_16)
    real(c_double), intent(in) :: t3(extent_16)
    real(c_double), intent(in) :: t4(extent_16)
    real(c_double), intent(in) :: t5(extent_16)
    real(c_double), intent(in) :: t6(extent_16)
    real(c_double), intent(in) :: t7(extent_16)
    real(c_double), intent(in) :: t8(extent_16)
    real(c_double), intent(in) :: t9(extent_16)
    real(c_double), intent(in) :: t10(extent_16)
    real(c_double), intent(in) :: t11(extent_16)
    real(c_double), intent(in) :: t12(extent_16)
    real(c_double), intent(in) :: t13(extent_16)
    real(c_double), intent(in) :: t14(extent_16)
    real(c_double), intent(in) :: t15(extent_16)
    real(c_double), intent(in) :: t16(extent_16)
    real(c_double), intent(in) :: t17(extent_16)
    real(c_double), intent(in) :: t18(extent_16)
    real(c_double), intent(in) :: t19(extent_16)
    real(c_double), intent(in) :: t20(extent_16)
    real(c_double), intent(out) :: t514(extent_16)
    real(c_double), intent(out) :: t520(extent_16)
    real(c_double), intent(out) :: t526(extent_16)
    real(c_double), intent(out) :: t210(extent_16)
    real(c_double), intent(out) :: t208(extent_16)
    real(c_double), intent(out) :: t131(extent_16)
    real(c_double), intent(out) :: t139(extent_16)
    real(c_double), intent(out) :: t121(extent_16)
    real(c_double), intent(out) :: t123(extent_16)
    real(c_double), intent(out) :: t21(extent_16)
    real(c_double), intent(out) :: t396(extent_16)
    real(c_double), intent(out) :: t410(extent_16)
    real(c_double), intent(out) :: t424(extent_16)
    real(c_double), intent(out) :: t438(extent_16)
    real(c_double), intent(out) :: t452(extent_16)
    real(c_double), intent(out) :: t466(extent_16)

    ! block entry
    call numerical_region_0(extent_1, extent_16, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t121, t123, t131, t139, t396, t410, t424, t438, t452, t466, t208, t210, t514, t526, t520)
    return
  end subroutine columnar_multifluid_rgb_step_control
  subroutine numerical_region_0(extent_1, extent_16, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t121, t123, t131, t139, t396, t410, t424, t438, t452, t466, t208, t210, t514, t526, t520) bind(C, name="numerical_region_0")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_int), intent(in), value :: extent_1
    integer(c_int), intent(in), value :: extent_16
    real(c_double), intent(in) :: t0(extent_16)
    real(c_double), intent(in) :: t1(extent_16)
    real(c_double), intent(in) :: t2(extent_16)
    real(c_double), intent(in) :: t3(extent_16)
    real(c_double), intent(in) :: t4(extent_16)
    real(c_double), intent(in) :: t5(extent_16)
    real(c_double), intent(in) :: t6(extent_16)
    real(c_double), intent(in) :: t7(extent_16)
    real(c_double), intent(in) :: t8(extent_16)
    real(c_double), intent(in) :: t9(extent_16)
    real(c_double), intent(in) :: t10(extent_16)
    real(c_double), intent(in) :: t11(extent_16)
    real(c_double), intent(in) :: t12(extent_16)
    real(c_double), intent(in) :: t13(extent_16)
    real(c_double), intent(in) :: t14(extent_16)
    real(c_double), intent(in) :: t15(extent_16)
    real(c_double), intent(in) :: t16(extent_16)
    real(c_double), intent(in) :: t17(extent_16)
    real(c_double), intent(in) :: t18(extent_16)
    real(c_double), intent(in) :: t19(extent_16)
    real(c_double), intent(in) :: t20(extent_16)
    real(c_double), intent(out) :: t21(extent_16)
    real(c_double), intent(out) :: t121(extent_16)
    real(c_double), intent(out) :: t123(extent_16)
    real(c_double), intent(out) :: t131(extent_16)
    real(c_double), intent(out) :: t139(extent_16)
    real(c_double), intent(out) :: t396(extent_16)
    real(c_double), intent(out) :: t410(extent_16)
    real(c_double), intent(out) :: t424(extent_16)
    real(c_double), intent(out) :: t438(extent_16)
    real(c_double), intent(out) :: t452(extent_16)
    real(c_double), intent(out) :: t466(extent_16)
    real(c_double), intent(out) :: t208(extent_16)
    real(c_double), intent(out) :: t210(extent_16)
    real(c_double), intent(out) :: t514(extent_16)
    real(c_double), intent(out) :: t526(extent_16)
    real(c_double), intent(out) :: t520(extent_16)
    real(c_double) :: t22(extent_16)
    real(c_double) :: t23(extent_1)
    real(c_double) :: t36(extent_1)
    real(c_double) :: t38(extent_1)
    real(c_double) :: t41(extent_1)
    real(c_double) :: t43(extent_1)
    real(c_double) :: t58(extent_16)
    real(c_double) :: t59(extent_16)
    real(c_double) :: t94(extent_1)
    real(c_double) :: t103(extent_1)
    real(c_double) :: t48(extent_1)
    real(c_double) :: t113(extent_1)
    real(c_double) :: t140(extent_1)
    real(c_double) :: t142(extent_1)
    real(c_double) :: t76(extent_1)
    real(c_double) :: t148(extent_1)
    real(c_double) :: t204(extent_1)
    real(c_double) :: t384(extent_1)
    real(c_double) :: t389(extent_1)
    real(c_double) :: t398(extent_1)
    real(c_double) :: t412(extent_1)
    real(c_double) :: t426(extent_1)
    real(c_double) :: t440(extent_1)
    real(c_double) :: t454(extent_1)
    real(c_double) :: t102(extent_1)
    real(c_double) :: t288(extent_16)
    real(c_double) :: t26(extent_16)
    real(c_double) :: t62(extent_16)
    real(c_double) :: t348(extent_1)
    real(c_double) :: t355(extent_1)
    real(c_double) :: t362(extent_1)
    real(c_double) :: t369(extent_1)
    real(c_double) :: t376(extent_1)
    real(c_double) :: t63(extent_1)
    real(c_double) :: t50(extent_16)
    real(c_double) :: t66(extent_16)
    real(c_double) :: t72(extent_1)
    real(c_double) :: t30(extent_1)
    real(c_double) :: t155(extent_1)
    real(c_double) :: t78(extent_16)
    real(c_double) :: t33(extent_16)
    real(c_double) :: t158(extent_1)
    real(c_double) :: t157(extent_1)
    real(c_double) :: t160(extent_16)
    real(c_double) :: t82(extent_1)
    real(c_double) :: t83
    real(c_double) :: t79(extent_16)
    real(c_double) :: t100(extent_1)
    real(c_double) :: t127(extent_1)
    real(c_double) :: t130(extent_1)
    real(c_double) :: t138(extent_1)
    real(c_double) :: t167(extent_16)
    real(c_double) :: t190(extent_1)
    real(c_double) :: t294(extent_1)
    real(c_double) :: t303(extent_1)
    real(c_double) :: t312(extent_1)
    real(c_double) :: t321(extent_1)
    real(c_double) :: t330(extent_1)
    real(c_double) :: t339(extent_1)
    real(c_double) :: t184(extent_16)
    real(c_double) :: t193(extent_16)
    real(c_double) :: t195(extent_1)
    real(c_double) :: t197(extent_1)
    real(c_double) :: t243(extent_1)
    real(c_double) :: t259(extent_1)
    real(c_double) :: t201(extent_1)
    real(c_double) :: t508(extent_16)
    real(c_double) :: t512(extent_1)
    real(c_double) :: t518(extent_1)
    real(c_double) :: t250(extent_1)
    real(c_double) :: t474(extent_16)
    real(c_double) :: t476(extent_1)
    real(c_double) :: t477(extent_16)
    real(c_double) :: t212(extent_1)
    real(c_double) :: t230(extent_16)
    real(c_double) :: t221(extent_16)
    real(c_double) :: t240(extent_1)
    real(c_double) :: t256(extent_1)
    real(c_double) :: t275(extent_1)
    real(c_double) :: t237(extent_1)
    real(c_double) :: t253(extent_1)
    real(c_double) :: t272(extent_1)
    real(c_double) :: t236(extent_1)
    real(c_double) :: t252(extent_1)
    real(c_double) :: t271(extent_1)

    ! block entry
    t21 = (t0 + t1)
    t22 = (t2 - t3)
    t23 = 2.0_c_double
    t36 = 0.61_c_double
    t38 = 0.83_c_double
    t41 = 0.37_c_double
    t43 = 0.29_c_double
    t58 = (t5 - t7)
    t59 = (t6 - t8)
    t94 = 0.0_c_double
    t94 = 0.0_c_double
    t103 = 5.0_c_double
    t48 = 0.72_c_double
    t113 = 3.5_c_double
    t48 = 0.72_c_double
    t140 = 0.42_c_double
    t142 = 0.34_c_double
    t76 = 0.18_c_double
    t148 = 0.35_c_double
    t204 = 8.0_c_double
    t384 = -0.05_c_double
    t389 = 2.8_c_double
    t398 = -0.054_c_double
    t389 = 2.8_c_double
    t412 = -0.058_c_double
    t389 = 2.8_c_double
    t426 = -0.062_c_double
    t389 = 2.8_c_double
    t440 = -0.066_c_double
    t389 = 2.8_c_double
    t454 = -0.07_c_double
    t389 = 2.8_c_double
    t102 = 0.1_c_double
    t102 = 0.1_c_double
    t140 = 0.42_c_double
    t288 = (t21 * t140(1))
    t26 = (((t4 * t23(1)) - t2) - t3)
    t62 = ((t58 * t58) + (t59 * t59))
    t348 = 1.0471975511965976_c_double
    t355 = 2.0943951023931953_c_double
    t362 = 3.141592653589793_c_double
    t369 = 4.1887902047863905_c_double
    t376 = 5.235987755982989_c_double
    t48 = 0.72_c_double
    t63 = 0.12_c_double
    t76 = 0.18_c_double
    t94 = 0.0_c_double
    t50 = (((t5 * t36(1)) + (t6 * t38(1))) + (sin(((t5 * t41(1)) - (t6 * t43(1)))) * t48(1)))
    t66 = sqrt((t62 + t63(1)))
    t72 = 11.045000000000002_c_double
    t94 = 0.0_c_double
    t94 = 0.0_c_double
    t94 = 0.0_c_double
    t94 = 0.0_c_double
    t94 = 0.0_c_double
    t94 = 0.0_c_double
    t30 = 1.0e-05_c_double
    t155 = 1.0_c_double
    t78 = (exp(((-t62) / t72(1))) / (t62 + t76(1)))
    t33 = sqrt((((t22 * t22) + (t26 * t26)) + t30(1)))
    t158 = 0.3_c_double
    t157 = 0.32_c_double
    t160 = ((min(max(((((t2 * t140(1)) + (t4 * t142(1))) + (t3 * t76(1))) + (t11 * t148(1))), t94(1)), t155(1)) * t158(1)) + t157(1))
    t82 = 1.0e-06_c_double
    t83 = (sum(t78) + t82(1))
    t79 = (t78 * ((cos(t50) * (t22 / t33)) + (sin(t50) * (t26 / t33))))
    t100 = 1.85_c_double
    t100 = 1.85_c_double
    t121 = (t9 + ((((((t5 * t94(1)) + (sum(((t58 / t66) * t79)) / t83)) * t100(1)) + ((t103(1) - t7) * t102(1))) - (t9 * t48(1))) * t1))
    t123 = (t10 + ((((((t6 * t94(1)) + (sum(((t59 / t66) * t79)) / t83)) * t100(1)) + ((t113(1) - t8) * t102(1))) - (t10 * t48(1))) * t1))
    t127 = 0.65_c_double
    t127 = 0.65_c_double
    t130 = 9.35_c_double
    t131 = min(max((t7 + (t121 * t1)), t127(1)), t130(1))
    t138 = 6.35_c_double
    t139 = min(max((t8 + (t123 * t1)), t127(1)), t138(1))
    t167 = (((t5 - t131) * (t5 - t131)) + ((t6 - t139) * (t6 - t139)))
    t94 = 0.0_c_double
    t94 = 0.0_c_double
    t190 = 3.6450000000000005_c_double
    t294 = 0.3872_c_double
    t303 = 0.4608_c_double
    t312 = 0.5408000000000001_c_double
    t321 = 0.6272000000000001_c_double
    t330 = 0.72_c_double
    t339 = 0.8192_c_double
    t184 = (min(max((t160 - abs((t5 - t131))), t94(1)), max((t160 - abs((t6 - t139))), t94(1))) / t160)
    t193 = exp(((-t167) / t190(1)))
    t195 = -0.42_c_double
    t197 = 0.22_c_double
    t243 = 54.0_c_double
    t259 = 30.0_c_double
    t201 = 20.0_c_double
    t508 = (t184 * t184)
    t155 = 1.0_c_double
    t512 = 245.0_c_double
    t155 = 1.0_c_double
    t518 = 252.0_c_double
    t155 = 1.0_c_double
    t250 = 255.0_c_double
    t155 = 1.0_c_double
    t396 = min(((t15 * exp((t1 * t384(1)))) + (((t1 * t389(1)) * exp(((-t167) / t294(1)))) * max(cos(t288), t94(1)))), t155(1))
    t155 = 1.0_c_double
    t410 = min(((t16 * exp((t1 * t398(1)))) + (((t1 * t389(1)) * exp(((-t167) / t303(1)))) * max(cos((t288 - t348(1))), t94(1)))), t155(1))
    t155 = 1.0_c_double
    t424 = min(((t17 * exp((t1 * t412(1)))) + (((t1 * t389(1)) * exp(((-t167) / t312(1)))) * max(cos((t288 - t355(1))), t94(1)))), t155(1))
    t155 = 1.0_c_double
    t438 = min(((t18 * exp((t1 * t426(1)))) + (((t1 * t389(1)) * exp(((-t167) / t321(1)))) * max(cos((t288 - t362(1))), t94(1)))), t155(1))
    t155 = 1.0_c_double
    t452 = min(((t19 * exp((t1 * t440(1)))) + (((t1 * t389(1)) * exp(((-t167) / t330(1)))) * max(cos((t288 - t369(1))), t94(1)))), t155(1))
    t155 = 1.0_c_double
    t466 = min(((t20 * exp((t1 * t454(1)))) + (((t1 * t389(1)) * exp(((-t167) / t339(1)))) * max(cos((t288 - t376(1))), t94(1)))), t155(1))
    t201 = 20.0_c_double
    t250 = 255.0_c_double
    t250 = 255.0_c_double
    t250 = 255.0_c_double
    t208 = (t13 + ((((((t193 * t195(1)) - ((t184 * t197(1)) * t184)) - t12) * t201(1)) - (t13 * t204(1))) * t1))
    t210 = (t12 + (t208 * t1))
    t155 = 1.0_c_double
    t82 = 1.0e-06_c_double
    t474 = max((((((t396 + t410) + t424) + t438) + t452) + t466), t82(1))
    t204 = 8.0_c_double
    t476 = 0.88_c_double
    t477 = min(t474, t476(1))
    t212 = 0.5_c_double
    t140 = 0.42_c_double
    t155 = 1.0_c_double
    t155 = 1.0_c_double
    t155 = 1.0_c_double
    t103 = 5.0_c_double
    t94 = 0.0_c_double
    t94 = 0.0_c_double
    t155 = 1.0_c_double
    t230 = min(max(((-t210) / t140(1)), t94(1)), t155(1))
    t155 = 1.0_c_double
    t221 = min(max((((t14 + t210) - t212(1)) / t103(1)), t94(1)), t155(1))
    t240 = 34.0_c_double
    t256 = 21.0_c_double
    t275 = 15.0_c_double
    t237 = 27.0_c_double
    t253 = 18.0_c_double
    t272 = 16.0_c_double
    t236 = 186.0_c_double
    t252 = 220.0_c_double
    t271 = 232.0_c_double
    t94 = 0.0_c_double
    t94 = 0.0_c_double
    t250 = 255.0_c_double
    t94 = 0.0_c_double
    t250 = 255.0_c_double
    t250 = 255.0_c_double
    t514 = ((((min(max(((((t221 * t237(1)) + t236(1)) - (t230 * t240(1))) + (t193 * t243(1))), t94(1)), t250(1)) * (t155(1) - t477)) + (((((t396 + t410) + t466) * t250(1)) / t474) * t477)) * (t155(1) - t508)) + (t508 * t512(1)))
    t526 = ((((min(max(((((t221 * t272(1)) + t271(1)) + (t230 * t275(1))) + (t193 * t201(1))), t94(1)), t250(1)) * (t155(1) - t477)) + (((((t438 + t452) + t466) * t250(1)) / t474) * t477)) * (t155(1) - t508)) + (t508 * t250(1)))
    t520 = ((((min(max((((((t221 * t253(1)) + t252(1)) - (t230 * t256(1))) + (t193 * t259(1))) + (min(abs(t208), t155(1)) * t204(1))), t94(1)), t250(1)) * (t155(1) - t477)) + (((((t410 + t424) + t438) * t250(1)) / t474) * t477)) * (t155(1) - t508)) + (t508 * t518(1)))
    return
  end subroutine numerical_region_0

end module kernel_fortran
