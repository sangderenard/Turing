module kernel_fortran
  use, intrinsic :: iso_c_binding
  implicit none
contains

  subroutine columnar_multifluid_rgb_step_numerical(extent_1, extent_16, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t1393, t1397, t1401, t976, t974, t301, t309, t291, t293, t457, t465, t447, t449, t603, t611, t593, t595, t875, t884, t893, t942, t949, t1228, t35, t1215, t1216, t1217, t1218, t1219, t1220) bind(C, name="columnar_multifluid_rgb_step_numerical")
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
    real(c_double), intent(in) :: t21(extent_16)
    real(c_double), intent(in) :: t22(extent_16)
    real(c_double), intent(in) :: t23(extent_16)
    real(c_double), intent(in) :: t24(extent_16)
    real(c_double), intent(in) :: t25(extent_16)
    real(c_double), intent(in) :: t26(extent_16)
    real(c_double), intent(in) :: t27(extent_16)
    real(c_double), intent(in) :: t28(extent_16)
    real(c_double), intent(in) :: t29(extent_16)
    real(c_double), intent(in) :: t30(extent_16)
    real(c_double), intent(in) :: t31(extent_16)
    real(c_double), intent(in) :: t32(extent_16)
    real(c_double), intent(in) :: t33(extent_16)
    real(c_double), intent(in) :: t34(extent_16)
    real(c_double), intent(out) :: t1393(extent_16)
    real(c_double), intent(out) :: t1397(extent_16)
    real(c_double), intent(out) :: t1401(extent_16)
    real(c_double), intent(out) :: t976(extent_16)
    real(c_double), intent(out) :: t974(extent_16)
    real(c_double), intent(out) :: t301(extent_16)
    real(c_double), intent(out) :: t309(extent_16)
    real(c_double), intent(out) :: t291(extent_16)
    real(c_double), intent(out) :: t293(extent_16)
    real(c_double), intent(out) :: t457(extent_16)
    real(c_double), intent(out) :: t465(extent_16)
    real(c_double), intent(out) :: t447(extent_16)
    real(c_double), intent(out) :: t449(extent_16)
    real(c_double), intent(out) :: t603(extent_16)
    real(c_double), intent(out) :: t611(extent_16)
    real(c_double), intent(out) :: t593(extent_16)
    real(c_double), intent(out) :: t595(extent_16)
    real(c_double), intent(out) :: t875(extent_16)
    real(c_double), intent(out) :: t884(extent_16)
    real(c_double), intent(out) :: t893(extent_16)
    real(c_double), intent(out) :: t942(extent_16)
    real(c_double), intent(out) :: t949(extent_16)
    real(c_double), intent(out) :: t1228(extent_16)
    real(c_double), intent(out) :: t35(extent_16)
    real(c_double), intent(out) :: t1215(extent_16)
    real(c_double), intent(out) :: t1216(extent_16)
    real(c_double), intent(out) :: t1217(extent_16)
    real(c_double), intent(out) :: t1218(extent_16)
    real(c_double), intent(out) :: t1219(extent_16)
    real(c_double), intent(out) :: t1220(extent_16)
    real(c_double) :: t36(extent_16)
    real(c_double) :: t72(extent_16)
    real(c_double) :: t73(extent_16)
    real(c_double) :: t74(extent_16)
    real(c_double) :: t150(extent_16)
    real(c_double) :: t151(extent_16)
    real(c_double) :: t206(extent_16)
    real(c_double) :: t203(extent_16)
    real(c_double) :: t204(extent_16)
    real(c_double) :: t205(extent_16)
    real(c_double) :: t217(extent_16)
    real(c_double) :: t218(extent_16)
    real(c_double) :: t310(extent_16)
    real(c_double) :: t311(extent_16)
    real(c_double) :: t366(extent_16)
    real(c_double) :: t367(extent_16)
    real(c_double) :: t373(extent_16)
    real(c_double) :: t374(extent_16)
    real(c_double) :: t466(extent_16)
    real(c_double) :: t467(extent_16)
    real(c_double) :: t519(extent_16)
    real(c_double) :: t520(extent_16)
    real(c_double) :: t901(extent_16)
    real(c_double) :: t903(extent_16)
    real(c_double) :: t905(extent_16)
    real(c_double) :: t1146(extent_16)
    real(c_double) :: t1148(extent_16)
    real(c_double) :: t1283(extent_16)
    real(c_double) :: t1284(extent_16)
    real(c_double) :: t80(extent_16)
    real(c_double) :: t86(extent_16)
    real(c_double) :: t92(extent_16)
    real(c_double) :: t955(extent_16)
    real(c_double) :: t1139(extent_16)
    real(c_double) :: t1144(extent_16)
    real(c_double) :: t40(extent_16)
    real(c_double) :: t94(extent_16)
    real(c_double) :: t96(extent_16)
    real(c_double) :: t98(extent_16)
    real(c_double) :: t104(extent_16)
    real(c_double) :: t154(extent_16)
    real(c_double) :: t314(extent_16)
    real(c_double) :: t470(extent_16)
    real(c_double) :: t908(extent_16)
    real(c_double) :: t911(extent_16)
    real(c_double) :: t211(extent_16)
    real(c_double) :: t216(extent_16)
    real(c_double) :: t372(extent_16)
    real(c_double) :: t64(extent_16)
    real(c_double) :: t225(extent_16)
    real(c_double) :: t381(extent_16)
    real(c_double) :: t527(extent_16)
    real(c_double) :: t1114(extent_16)
    real(c_double) :: t66(extent_16)
    real(c_double) :: t68(extent_16)
    real(c_double) :: t920(extent_16)
    real(c_double) :: t929(extent_16)
    real(c_double) :: t1124(extent_16)
    real(c_double) :: t1134(extent_16)
    real(c_double) :: t1296(extent_16)
    real(c_double) :: t1349(extent_16)
    real(c_double) :: t330(extent_16)
    real(c_double) :: t1358(extent_16)
    real(c_double) :: t47(extent_16)
    real(c_double) :: t170(extent_16)
    real(c_double) :: t486(extent_16)
    real(c_double) :: t1313(extent_16)
    real(c_double) :: t1360(extent_16)
    real(c_double) :: t1362(extent_16)
    real(c_double) :: t49(extent_16)
    real(c_double) :: t48(extent_16)
    real(c_double) :: t632(extent_16)
    real(c_double) :: t1326(extent_16)
    real(c_double) :: t1214(extent_16)
    real(c_double) :: t1328(extent_16)
    real(c_double) :: t71(extent_16)
    real(c_double) :: t132(extent_16)
    real(c_double) :: t149(extent_16)
    real(c_double) :: t190(extent_16)
    real(c_double) :: t353(extent_16)
    real(c_double) :: t506(extent_16)
    real(c_double) :: t646(extent_16)
    real(c_double) :: t639(extent_16)
    real(c_double) :: t653(extent_16)
    real(c_double) :: t749(extent_16)
    real(c_double) :: t767(extent_16)
    real(c_double) :: t758(extent_16)
    real(c_double) :: t708(extent_16)
    real(c_double) :: t860(extent_16)
    real(c_double) :: t863(extent_16)
    real(c_double) :: t866(extent_16)
    real(c_double) :: t740(extent_16)
    real(c_double) :: t1387(extent_16)
    real(c_double) :: t794(extent_1)
    real(c_double) :: t800(extent_1)
    real(c_double) :: t806(extent_1)
    real(c_double) :: t1155(extent_16)
    real(c_double) :: t1162(extent_16)
    real(c_double) :: t1169(extent_16)
    real(c_double) :: t1176(extent_16)
    real(c_double) :: t1183(extent_16)
    real(c_double) :: t1190(extent_16)
    real(c_double) :: t1389(extent_16)
    real(c_double) :: t797(extent_16)
    real(c_double) :: t803(extent_16)
    real(c_double) :: t809(extent_16)
    real(c_double) :: t1299(extent_16)
    real(c_double) :: t1236(extent_16)
    real(c_double) :: t1239(extent_16)
    real(c_double) :: t1256(extent_16)
    real(c_double) :: t1268(extent_16)
    real(c_double) :: t1270(extent_16)
    real(c_double) :: t996(extent_16)
    real(c_double) :: t987(extent_16)

    ! block entry
    t35 = (t0 + t1)
    t36 = (t2 - t3)
    t72 = (t7 + t8)
    t73 = (t9 + t10)
    t74 = (t11 + t12)
    t150 = (t5 - t17)
    t151 = (t6 - t18)
    t206 = (t18 - t22)
    t203 = (t17 - t19)
    t204 = (t18 - t20)
    t205 = (t17 - t21)
    t217 = (5.0_c_double - t17)
    t218 = (3.45_c_double - t18)
    t310 = (t5 - t19)
    t311 = (t6 - t20)
    t366 = (t19 - t21)
    t367 = (t20 - t22)
    t373 = (5.0_c_double - t19)
    t374 = (3.45_c_double - t20)
    t466 = (t5 - t21)
    t467 = (t6 - t22)
    t519 = (5.0_c_double - t21)
    t520 = (3.45_c_double - t22)
    t901 = (t5 - 1.15_c_double)
    t903 = (t6 - 5.75_c_double)
    t905 = (t5 - 8.85_c_double)
    t1146 = (t1 * 3.2_c_double)
    t1148 = (t1 * 2.2_c_double)
    t1283 = (t5 - 5.0_c_double)
    t1284 = (t6 - 3.45_c_double)
    t80 = min(max(t13, 0.0_c_double), 1.0_c_double)
    t86 = min(max(t14, 0.0_c_double), 1.0_c_double)
    t92 = min(max(t15, 0.0_c_double), 1.0_c_double)
    t955 = min(max(t31, 0.0_c_double), 0.42_c_double)
    t1139 = exp((t1 * -0.095_c_double))
    t1144 = exp((t1 * -0.072_c_double))
    t40 = (((t4 * 2.0_c_double) - t2) - t3)
    t94 = (1.0_c_double - t80)
    t96 = (1.0_c_double - t86)
    t98 = (1.0_c_double - t92)
    t104 = real(floor((t35 * 0.08_c_double)), c_double)
    t154 = ((t150 * t150) + (t151 * t151))
    t314 = ((t310 * t310) + (t311 * t311))
    t470 = ((t466 * t466) + (t467 * t467))
    t908 = ((t901 * t901) + (t903 * t903))
    t911 = ((t905 * t905) + (t903 * t903))
    t211 = (((t203 * t203) + (t204 * t204)) + 0.18_c_double)
    t216 = (((t205 * t205) + (t206 * t206)) + 0.18_c_double)
    t372 = (((t366 * t366) + (t367 * t367)) + 0.18_c_double)
    t64 = (((t5 * 0.61_c_double) + (t6 * 0.83_c_double)) + (sin(((t5 * 0.37_c_double) - (t6 * 0.29_c_double))) * 0.72_c_double))
    t225 = sqrt((((t217 * t217) + (t218 * t218)) + 0.08_c_double))
    t381 = sqrt((((t373 * t373) + (t374 * t374)) + 0.08_c_double))
    t527 = sqrt((((t519 * t519) + (t520 * t520)) + 0.08_c_double))
    t1114 = ((sin((t35 * 2.11_c_double)) * 0.22_c_double) + 0.78_c_double)
    t66 = cos(t64)
    t68 = sin(t64)
    t920 = exp(((-t908) / 0.4608_c_double))
    t929 = exp(((-t911) / 0.6728_c_double))
    t1124 = ((sin(((t35 * 1.91_c_double) + 2.1_c_double)) * 0.22_c_double) + 0.78_c_double)
    t1134 = ((sin(((t35 * 2.27_c_double) + 4.2_c_double)) * 0.22_c_double) + 0.78_c_double)
    t1296 = exp(((-((t1283 * t1283) + (t1284 * t1284))) / 0.23120000000000004_c_double))
    t1349 = exp(((-t908) / 0.1058_c_double))
    t330 = (exp(((-t314) / 4.805000000000001_c_double)) / (t314 + 0.14_c_double))
    t1358 = exp(((-t911) / 0.1058_c_double))
    t47 = sqrt((((t36 * t36) + (t40 * t40)) + 1.0e-05_c_double))
    t170 = (exp(((-t154) / 4.805000000000001_c_double)) / (t154 + 0.14_c_double))
    t486 = (exp(((-t470) / 4.805000000000001_c_double)) / (t470 + 0.14_c_double))
    t1313 = (1.0_c_double - t1296)
    t1360 = (1.0_c_double - t1349)
    t1362 = (1.0_c_double - t1358)
    t49 = (t40 / t47)
    t48 = (t36 / t47)
    t632 = ((min(max(((((t2 * 0.42_c_double) + (t4 * 0.34_c_double)) + (t3 * 0.18_c_double)) + (t29 * 0.35_c_double)), 0.0_c_double), 1.0_c_double) * 0.12_c_double) + 0.18_c_double)
    t1326 = min((t929 * t955), 0.6_c_double)
    t1214 = (1.0_c_double - min(((t1 * 0.12_c_double) * t920), 0.08_c_double))
    t1328 = (1.0_c_double - t1326)
    t71 = ((t66 * t48) + (t68 * t49))
    t132 = (max(((sin((((t5 * 12.9898_c_double) + (t6 * 78.233_c_double)) + (t104 * 37.719_c_double))) * cos((((t5 * 39.3467_c_double) - (t6 * 11.135_c_double)) + (t104 * 19.913_c_double)))) - 0.72_c_double), 0.0_c_double) / 0.28_c_double)
    t149 = min(max(((t16 * exp((t1 * -0.003_c_double))) + ((t1 * 0.032_c_double) * (t132 * t132))), 0.0_c_double), 1.0_c_double)
    t190 = ((t170 * (((((t94 * 1.2_c_double) * t74) - (t72 * 0.3_c_double)) + ((t94 * 1.65_c_double) * t149)) + (t71 * 0.18_c_double))) / (sqrt((t154 + 0.08_c_double)) * (sum(t170) + 1.0e-06_c_double)))
    t353 = ((t330 * (((((t96 * 1.2_c_double) * t72) - (t73 * 0.3_c_double)) + ((t96 * 1.65_c_double) * t149)) + (((t68 * t48) - (t66 * t49)) * 0.18_c_double))) / (sqrt((t314 + 0.08_c_double)) * (sum(t330) + 1.0e-06_c_double)))
    t506 = ((t486 * (((((t98 * 1.2_c_double) * t73) - (t74 * 0.3_c_double)) + ((t98 * 1.65_c_double) * t149)) - (t71 * 0.18_c_double))) / (sqrt((t470 + 0.08_c_double)) * (sum(t486) + 1.0e-06_c_double)))
    t291 = (t23 + ((((((((((t5 * 0.0_c_double) + sum((t150 * t190))) * 2.35_c_double) + (((t80 * 1.85_c_double) * t217) / t225)) + (cos(((t35 * 1.71_c_double) + 0.2_c_double)) * 0.58_c_double)) + ((t203 * 0.3_c_double) / t211)) + ((t205 * 0.3_c_double) / t216)) + ((5.0_c_double - t17) * 0.08_c_double)) - (t23 * 0.54_c_double)) * t1))
    t293 = (t24 + ((((((((((t6 * 0.0_c_double) + sum((t151 * t190))) * 2.35_c_double) + (((t80 * 1.85_c_double) * t218) / t225)) + (sin(((t35 * 1.37_c_double) + 1.1_c_double)) * 0.58_c_double)) + ((t204 * 0.3_c_double) / t211)) + ((t206 * 0.3_c_double) / t216)) + ((3.5_c_double - t18) * 0.08_c_double)) - (t24 * 0.54_c_double)) * t1))
    t447 = (t25 + ((((((((((t5 * 0.0_c_double) + sum((t310 * t353))) * 2.35_c_double) + (((t86 * 1.85_c_double) * t373) / t381)) + (cos(((t35 * 1.63_c_double) + 2.3_c_double)) * 0.58_c_double)) - ((t203 * 0.3_c_double) / t211)) + ((t366 * 0.3_c_double) / t372)) + ((5.0_c_double - t19) * 0.08_c_double)) - (t25 * 0.54_c_double)) * t1))
    t449 = (t26 + ((((((((((t6 * 0.0_c_double) + sum((t311 * t353))) * 2.35_c_double) + (((t86 * 1.85_c_double) * t374) / t381)) + (sin(((t35 * 1.43_c_double) + 2.8_c_double)) * 0.58_c_double)) - ((t204 * 0.3_c_double) / t211)) + ((t367 * 0.3_c_double) / t372)) + ((3.5_c_double - t20) * 0.08_c_double)) - (t26 * 0.54_c_double)) * t1))
    t593 = (t27 + ((((((((((t5 * 0.0_c_double) + sum((t466 * t506))) * 2.35_c_double) + (((t92 * 1.85_c_double) * t519) / t527)) + (cos(((t35 * 1.79_c_double) + 4.2_c_double)) * 0.58_c_double)) - ((t205 * 0.3_c_double) / t216)) - ((t366 * 0.3_c_double) / t372)) + ((5.0_c_double - t21) * 0.08_c_double)) - (t27 * 0.54_c_double)) * t1))
    t595 = (t28 + ((((((((((t6 * 0.0_c_double) + sum((t467 * t506))) * 2.35_c_double) + (((t92 * 1.85_c_double) * t520) / t527)) + (sin(((t35 * 1.31_c_double) + 5.0_c_double)) * 0.58_c_double)) - ((t206 * 0.3_c_double) / t216)) - ((t367 * 0.3_c_double) / t372)) + ((3.5_c_double - t22) * 0.08_c_double)) - (t28 * 0.54_c_double)) * t1))
    t301 = min(max((t17 + (t291 * t1)), 0.65_c_double), 9.35_c_double)
    t309 = min(max((t18 + (t293 * t1)), 0.65_c_double), 6.35_c_double)
    t457 = min(max((t19 + (t447 * t1)), 0.65_c_double), 9.35_c_double)
    t465 = min(max((t20 + (t449 * t1)), 0.65_c_double), 6.35_c_double)
    t603 = min(max((t21 + (t593 * t1)), 0.65_c_double), 9.35_c_double)
    t611 = min(max((t22 + (t595 * t1)), 0.65_c_double), 6.35_c_double)
    t646 = (((t5 - t457) * (t5 - t457)) + ((t6 - t465) * (t6 - t465)))
    t639 = (((t5 - t301) * (t5 - t301)) + ((t6 - t309) * (t6 - t309)))
    t653 = (((t5 - t603) * (t5 - t603)) + ((t6 - t611) * (t6 - t611)))
    t749 = exp(((-t639) / 0.18_c_double))
    t767 = exp(((-t653) / 0.18_c_double))
    t758 = exp(((-t646) / 0.18_c_double))
    t708 = max(max((min(max((t632 - abs((t5 - t301))), 0.0_c_double), max((t632 - abs((t6 - t309))), 0.0_c_double)) / t632), (min(max((t632 - abs((t5 - t457))), 0.0_c_double), max((t632 - abs((t6 - t465))), 0.0_c_double)) / t632)), (min(max((t632 - abs((t5 - t603))), 0.0_c_double), max((t632 - abs((t6 - t611))), 0.0_c_double)) / t632))
    t860 = ((t80 * exp(((-(((t301 - 5.0_c_double) * (t301 - 5.0_c_double)) + ((t309 - 3.45_c_double) * (t309 - 3.45_c_double)))) / 0.32000000000000006_c_double))) * 2.4_c_double)
    t863 = ((t86 * exp(((-(((t457 - 5.0_c_double) * (t457 - 5.0_c_double)) + ((t465 - 3.45_c_double) * (t465 - 3.45_c_double)))) / 0.32000000000000006_c_double))) * 2.4_c_double)
    t866 = ((t92 * exp(((-(((t603 - 5.0_c_double) * (t603 - 5.0_c_double)) + ((t611 - 3.45_c_double) * (t611 - 3.45_c_double)))) / 0.32000000000000006_c_double))) * 2.4_c_double)
    t740 = min(((exp(((-t639) / 1.2168_c_double)) + exp(((-t646) / 1.2168_c_double))) + exp(((-t653) / 1.2168_c_double))), 1.0_c_double)
    t1387 = (t708 * t708)
    t794 = min((sum((t149 * t749)) / (sum(t749) + 1.0e-06_c_double)), 0.7_c_double)
    t800 = min((sum((t149 * t758)) / (sum(t758) + 1.0e-06_c_double)), 0.7_c_double)
    t806 = min((sum((t149 * t767)) / (sum(t767) + 1.0e-06_c_double)), 0.7_c_double)
    t1155 = min(((t7 * t1139) + ((t1146 * exp(((-t639) / 0.1152_c_double))) * t1114)), 1.0_c_double)
    t1162 = min(((t8 * t1144) + ((t1148 * exp(((-t639) / 0.2888_c_double))) * t1114)), 1.0_c_double)
    t1169 = min(((t9 * t1139) + ((t1146 * exp(((-t646) / 0.1152_c_double))) * t1124)), 1.0_c_double)
    t1176 = min(((t10 * t1144) + ((t1148 * exp(((-t646) / 0.2888_c_double))) * t1124)), 1.0_c_double)
    t1183 = min(((t11 * t1139) + ((t1146 * exp(((-t653) / 0.1152_c_double))) * t1134)), 1.0_c_double)
    t1190 = min(((t12 * t1144) + ((t1148 * exp(((-t653) / 0.2888_c_double))) * t1134)), 1.0_c_double)
    t1389 = (1.0_c_double - t1387)
    t1219 = (t1183 * t1214)
    t1220 = (t1190 * t1214)
    t1218 = (t1176 * t1214)
    t1216 = (t1162 * t1214)
    t1217 = (t1169 * t1214)
    t1215 = (t1155 * t1214)
    t797 = ((t94 * t794(1)) * 1.35_c_double)
    t803 = ((t96 * t800(1)) * 1.35_c_double)
    t809 = ((t98 * t806(1)) * 1.35_c_double)
    t949 = max((t30 + (t1 * ((t860 + t863) + t866))), 0.0_c_double)
    t1299 = min(t949, 1.0_c_double)
    t875 = min(max((t80 + (t1 * (t797 - t860))), 0.0_c_double), 1.0_c_double)
    t884 = min(max((t86 + (t1 * (t803 - t863))), 0.0_c_double), 1.0_c_double)
    t893 = min(max((t92 + (t1 * (t809 - t866))), 0.0_c_double), 1.0_c_double)
    t974 = (t33 + (((((((t740 * -0.42_c_double) - ((t708 * 0.22_c_double) * t708)) + ((t955 * 0.16_c_double) * t929)) - t32) * 20.0_c_double) - (t33 * 4.6_c_double)) * t1))
    t1236 = max((((((t1215 + t1216) + t1217) + t1218) + t1219) + t1220), 1.0e-06_c_double)
    t1239 = min(t1236, 0.88_c_double)
    t976 = (t32 + (t974 * t1))
    t1256 = (1.0_c_double - t1239)
    t942 = min((max((t149 - (t1 * (((t749 * t797) + (t758 * t803)) + (t767 * t809)))), 0.0_c_double) * (1.0_c_double - ((t1 * 0.1_c_double) * t920))), 1.0_c_double)
    t1268 = min(t942, 0.76_c_double)
    t1228 = max(((t31 + (((t1 * 0.12_c_double) * sum(((((((t1155 + t1162) + t1169) + t1176) + t1183) + t1190) * t920))) / (sum(t920) + 1.0e-06_c_double))) - ((t1 * 0.3_c_double) * t955)), 0.0_c_double)
    t1270 = (1.0_c_double - t1268)
    t996 = min(max(((-t976) / 0.42_c_double), 0.0_c_double), 1.0_c_double)
    t987 = min(max((((t34 + t976) - 0.5_c_double) / 5.0_c_double), 0.0_c_double), 1.0_c_double)
    t1397 = ((((((((((((((min(max((((((t987 * 18.0_c_double) + 220.0_c_double) - (t996 * 21.0_c_double)) + (t740 * 30.0_c_double)) + (min(abs(t974), 1.0_c_double) * 8.0_c_double)), 0.0_c_double), 255.0_c_double) * t1256) + (((((t1216 + t1217) + t1218) * 255.0_c_double) / t1236) * t1239)) * t1270) + (t1268 * 181.0_c_double)) * t1313) + (((t1299 * 42.0_c_double) + 72.0_c_double) * t1296)) * t1328) + (t1326 * 249.0_c_double)) * t1360) + (t1349 * 83.0_c_double)) * t1362) + (t1358 * 245.0_c_double)) * t1389) + (t1387 * 252.0_c_double))
    t1401 = ((((((((((((((min(max(((((t987 * 16.0_c_double) + 232.0_c_double) + (t996 * 15.0_c_double)) + (t740 * 20.0_c_double)), 0.0_c_double), 255.0_c_double) * t1256) + (((((t1218 + t1219) + t1220) * 255.0_c_double) / t1236) * t1239)) * t1270) + (t1268 * 62.0_c_double)) * t1313) + (((t1299 * 24.0_c_double) + 48.0_c_double) * t1296)) * t1328) + (t1326 * 255.0_c_double)) * t1360) + (t1349 * 103.0_c_double)) * t1362) + (t1358 * 252.0_c_double)) * t1389) + (t1387 * 255.0_c_double))
    t1393 = ((((((((((((((min(max(((((t987 * 27.0_c_double) + 186.0_c_double) - (t996 * 34.0_c_double)) + (t740 * 54.0_c_double)), 0.0_c_double), 255.0_c_double) * t1256) + (((((t1215 + t1216) + t1220) * 255.0_c_double) / t1236) * t1239)) * t1270) + (t1268 * 226.0_c_double)) * t1313) + (((t1299 * 58.0_c_double) + 102.0_c_double) * t1296)) * t1328) + (t1326 * 218.0_c_double)) * t1360) + (t1349 * 53.0_c_double)) * t1362) + (t1358 * 205.0_c_double)) * t1389) + (t1387 * 245.0_c_double))
    return
  end subroutine columnar_multifluid_rgb_step_numerical
  subroutine columnar_multifluid_rgb_step(extent_1, extent_16, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t1393, t1397, t1401, t976, t974, t301, t309, t291, t293, t457, t465, t447, t449, t603, t611, t593, t595, t875, t884, t893, t942, t949, t1228, t35, t1215, t1216, t1217, t1218, t1219, t1220) bind(C, name="columnar_multifluid_rgb_step")
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
    real(c_double), intent(in) :: t21(extent_16)
    real(c_double), intent(in) :: t22(extent_16)
    real(c_double), intent(in) :: t23(extent_16)
    real(c_double), intent(in) :: t24(extent_16)
    real(c_double), intent(in) :: t25(extent_16)
    real(c_double), intent(in) :: t26(extent_16)
    real(c_double), intent(in) :: t27(extent_16)
    real(c_double), intent(in) :: t28(extent_16)
    real(c_double), intent(in) :: t29(extent_16)
    real(c_double), intent(in) :: t30(extent_16)
    real(c_double), intent(in) :: t31(extent_16)
    real(c_double), intent(in) :: t32(extent_16)
    real(c_double), intent(in) :: t33(extent_16)
    real(c_double), intent(in) :: t34(extent_16)
    real(c_double), intent(out) :: t1393(extent_16)
    real(c_double), intent(out) :: t1397(extent_16)
    real(c_double), intent(out) :: t1401(extent_16)
    real(c_double), intent(out) :: t976(extent_16)
    real(c_double), intent(out) :: t974(extent_16)
    real(c_double), intent(out) :: t301(extent_16)
    real(c_double), intent(out) :: t309(extent_16)
    real(c_double), intent(out) :: t291(extent_16)
    real(c_double), intent(out) :: t293(extent_16)
    real(c_double), intent(out) :: t457(extent_16)
    real(c_double), intent(out) :: t465(extent_16)
    real(c_double), intent(out) :: t447(extent_16)
    real(c_double), intent(out) :: t449(extent_16)
    real(c_double), intent(out) :: t603(extent_16)
    real(c_double), intent(out) :: t611(extent_16)
    real(c_double), intent(out) :: t593(extent_16)
    real(c_double), intent(out) :: t595(extent_16)
    real(c_double), intent(out) :: t875(extent_16)
    real(c_double), intent(out) :: t884(extent_16)
    real(c_double), intent(out) :: t893(extent_16)
    real(c_double), intent(out) :: t942(extent_16)
    real(c_double), intent(out) :: t949(extent_16)
    real(c_double), intent(out) :: t1228(extent_16)
    real(c_double), intent(out) :: t35(extent_16)
    real(c_double), intent(out) :: t1215(extent_16)
    real(c_double), intent(out) :: t1216(extent_16)
    real(c_double), intent(out) :: t1217(extent_16)
    real(c_double), intent(out) :: t1218(extent_16)
    real(c_double), intent(out) :: t1219(extent_16)
    real(c_double), intent(out) :: t1220(extent_16)

    ! block entry
    call numerical_region_0(extent_1, extent_16, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t291, t293, t447, t449, t593, t595, t301, t309, t457, t465, t603, t611, t1215, t1216, t1217, t1218, t1219, t1220, t949, t875, t884, t893, t974, t976, t942, t1228, t1393, t1401, t1397)
    return
  end subroutine columnar_multifluid_rgb_step
  subroutine numerical_region_0(extent_1, extent_16, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t291, t293, t447, t449, t593, t595, t301, t309, t457, t465, t603, t611, t1215, t1216, t1217, t1218, t1219, t1220, t949, t875, t884, t893, t974, t976, t942, t1228, t1393, t1401, t1397) bind(C, name="numerical_region_0")
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
    real(c_double), intent(in) :: t21(extent_16)
    real(c_double), intent(in) :: t22(extent_16)
    real(c_double), intent(in) :: t23(extent_16)
    real(c_double), intent(in) :: t24(extent_16)
    real(c_double), intent(in) :: t25(extent_16)
    real(c_double), intent(in) :: t26(extent_16)
    real(c_double), intent(in) :: t27(extent_16)
    real(c_double), intent(in) :: t28(extent_16)
    real(c_double), intent(in) :: t29(extent_16)
    real(c_double), intent(in) :: t30(extent_16)
    real(c_double), intent(in) :: t31(extent_16)
    real(c_double), intent(in) :: t32(extent_16)
    real(c_double), intent(in) :: t33(extent_16)
    real(c_double), intent(in) :: t34(extent_16)
    real(c_double), intent(out) :: t35(extent_16)
    real(c_double), intent(out) :: t291(extent_16)
    real(c_double), intent(out) :: t293(extent_16)
    real(c_double), intent(out) :: t447(extent_16)
    real(c_double), intent(out) :: t449(extent_16)
    real(c_double), intent(out) :: t593(extent_16)
    real(c_double), intent(out) :: t595(extent_16)
    real(c_double), intent(out) :: t301(extent_16)
    real(c_double), intent(out) :: t309(extent_16)
    real(c_double), intent(out) :: t457(extent_16)
    real(c_double), intent(out) :: t465(extent_16)
    real(c_double), intent(out) :: t603(extent_16)
    real(c_double), intent(out) :: t611(extent_16)
    real(c_double), intent(out) :: t1215(extent_16)
    real(c_double), intent(out) :: t1216(extent_16)
    real(c_double), intent(out) :: t1217(extent_16)
    real(c_double), intent(out) :: t1218(extent_16)
    real(c_double), intent(out) :: t1219(extent_16)
    real(c_double), intent(out) :: t1220(extent_16)
    real(c_double), intent(out) :: t949(extent_16)
    real(c_double), intent(out) :: t875(extent_16)
    real(c_double), intent(out) :: t884(extent_16)
    real(c_double), intent(out) :: t893(extent_16)
    real(c_double), intent(out) :: t974(extent_16)
    real(c_double), intent(out) :: t976(extent_16)
    real(c_double), intent(out) :: t942(extent_16)
    real(c_double), intent(out) :: t1228(extent_16)
    real(c_double), intent(out) :: t1393(extent_16)
    real(c_double), intent(out) :: t1401(extent_16)
    real(c_double), intent(out) :: t1397(extent_16)
    real(c_double) :: t36(extent_16)
    real(c_double) :: t37(extent_1)
    real(c_double) :: t50(extent_1)
    real(c_double) :: t52(extent_1)
    real(c_double) :: t55(extent_1)
    real(c_double) :: t57(extent_1)
    real(c_double) :: t72(extent_16)
    real(c_double) :: t73(extent_16)
    real(c_double) :: t74(extent_16)
    real(c_double) :: t76(extent_1)
    real(c_double) :: t105(extent_1)
    real(c_double) :: t107(extent_1)
    real(c_double) :: t115(extent_1)
    real(c_double) :: t117(extent_1)
    real(c_double) :: t135(extent_1)
    real(c_double) :: t140(extent_1)
    real(c_double) :: t150(extent_16)
    real(c_double) :: t151(extent_16)
    real(c_double) :: t206(extent_16)
    real(c_double) :: t203(extent_16)
    real(c_double) :: t204(extent_16)
    real(c_double) :: t205(extent_16)
    real(c_double) :: t99(extent_1)
    real(c_double) :: t217(extent_16)
    real(c_double) :: t100(extent_1)
    real(c_double) :: t218(extent_16)
    real(c_double) :: t255(extent_1)
    real(c_double) :: t283(extent_1)
    real(c_double) :: t310(extent_16)
    real(c_double) :: t311(extent_16)
    real(c_double) :: t366(extent_16)
    real(c_double) :: t367(extent_16)
    real(c_double) :: t373(extent_16)
    real(c_double) :: t374(extent_16)
    real(c_double) :: t466(extent_16)
    real(c_double) :: t467(extent_16)
    real(c_double) :: t519(extent_16)
    real(c_double) :: t520(extent_16)
    real(c_double) :: t612(extent_1)
    real(c_double) :: t614(extent_1)
    real(c_double) :: t181(extent_1)
    real(c_double) :: t620(extent_1)
    real(c_double) :: t900(extent_1)
    real(c_double) :: t901(extent_16)
    real(c_double) :: t902(extent_1)
    real(c_double) :: t903(extent_16)
    real(c_double) :: t904(extent_1)
    real(c_double) :: t905(extent_16)
    real(c_double) :: t935(extent_1)
    real(c_double) :: t970(extent_1)
    real(c_double) :: t1136(extent_1)
    real(c_double) :: t1141(extent_1)
    real(c_double) :: t1145(extent_1)
    real(c_double) :: t1146(extent_16)
    real(c_double) :: t1147(extent_1)
    real(c_double) :: t1148(extent_16)
    real(c_double) :: t630(extent_1)
    real(c_double) :: t174(extent_1)
    real(c_double) :: t1283(extent_16)
    real(c_double) :: t1284(extent_16)
    real(c_double) :: t79(extent_1)
    real(c_double) :: t80(extent_16)
    real(c_double) :: t86(extent_16)
    real(c_double) :: t92(extent_16)
    real(c_double) :: t101(extent_1)
    real(c_double) :: t234(extent_1)
    real(c_double) :: t266(extent_1)
    real(c_double) :: t390(extent_1)
    real(c_double) :: t422(extent_1)
    real(c_double) :: t536(extent_1)
    real(c_double) :: t568(extent_1)
    real(c_double) :: t955(extent_16)
    real(c_double) :: t1109(extent_1)
    real(c_double) :: t1117(extent_1)
    real(c_double) :: t1127(extent_1)
    real(c_double) :: t1139(extent_16)
    real(c_double) :: t1144(extent_16)
    real(c_double) :: t40(extent_16)
    real(c_double) :: t94(extent_16)
    real(c_double) :: t96(extent_16)
    real(c_double) :: t98(extent_16)
    real(c_double) :: t104(extent_16)
    real(c_double) :: t154(extent_16)
    real(c_double) :: t228(extent_1)
    real(c_double) :: t236(extent_1)
    real(c_double) :: t268(extent_1)
    real(c_double) :: t314(extent_16)
    real(c_double) :: t392(extent_1)
    real(c_double) :: t424(extent_1)
    real(c_double) :: t470(extent_16)
    real(c_double) :: t538(extent_1)
    real(c_double) :: t908(extent_16)
    real(c_double) :: t911(extent_16)
    real(c_double) :: t963(extent_1)
    real(c_double) :: t1119(extent_1)
    real(c_double) :: t62(extent_1)
    real(c_double) :: t110(extent_1)
    real(c_double) :: t120(extent_1)
    real(c_double) :: t168(extent_1)
    real(c_double) :: t171(extent_1)
    real(c_double) :: t177(extent_1)
    real(c_double) :: t211(extent_16)
    real(c_double) :: t216(extent_16)
    real(c_double) :: t372(extent_16)
    real(c_double) :: t959(extent_1)
    real(c_double) :: t64(extent_16)
    real(c_double) :: t164(extent_1)
    real(c_double) :: t225(extent_16)
    real(c_double) :: t233(extent_1)
    real(c_double) :: t324(extent_1)
    real(c_double) :: t381(extent_16)
    real(c_double) :: t480(extent_1)
    real(c_double) :: t527(extent_16)
    real(c_double) :: t917(extent_1)
    real(c_double) :: t926(extent_1)
    real(c_double) :: t711(extent_1)
    real(c_double) :: t1114(extent_16)
    real(c_double) :: t1293(extent_1)
    real(c_double) :: t1346(extent_1)
    real(c_double) :: t1355(extent_1)
    real(c_double) :: t44(extent_1)
    real(c_double) :: t66(extent_16)
    real(c_double) :: t68(extent_16)
    real(c_double) :: t920(extent_16)
    real(c_double) :: t929(extent_16)
    real(c_double) :: t1124(extent_16)
    real(c_double) :: t1134(extent_16)
    real(c_double) :: t1296(extent_16)
    real(c_double) :: t1349(extent_16)
    real(c_double) :: t330(extent_16)
    real(c_double) :: t1358(extent_16)
    real(c_double) :: t47(extent_16)
    real(c_double) :: t170(extent_16)
    real(c_double) :: t486(extent_16)
    real(c_double) :: t1313(extent_16)
    real(c_double) :: t1360(extent_16)
    real(c_double) :: t1362(extent_16)
    real(c_double) :: t1364(extent_1)
    real(c_double) :: t1368(extent_1)
    real(c_double) :: t1372(extent_1)
    real(c_double) :: t1376(extent_1)
    real(c_double) :: t1380(extent_1)
    real(c_double) :: t1384(extent_1)
    real(c_double) :: t49(extent_16)
    real(c_double) :: t48(extent_16)
    real(c_double) :: t632(extent_16)
    real(c_double) :: t187(extent_1)
    real(c_double) :: t1325(extent_1)
    real(c_double) :: t1326(extent_16)
    real(c_double) :: t1214(extent_16)
    real(c_double) :: t1328(extent_16)
    real(c_double) :: t1330(extent_1)
    real(c_double) :: t1334(extent_1)
    real(c_double) :: t1016(extent_1)
    real(c_double) :: t71(extent_16)
    real(c_double) :: t131(extent_1)
    real(c_double) :: t132(extent_16)
    real(c_double) :: t149(extent_16)
    real(c_double) :: t190(extent_16)
    real(c_double) :: t353(extent_16)
    real(c_double) :: t506(extent_16)
    real(c_double) :: t226(extent_1)
    real(c_double) :: t297(extent_1)
    real(c_double) :: t300(extent_1)
    real(c_double) :: t308(extent_1)
    real(c_double) :: t646(extent_16)
    real(c_double) :: t639(extent_16)
    real(c_double) :: t653(extent_16)
    real(c_double) :: t714(extent_1)
    real(c_double) :: t723(extent_1)
    real(c_double) :: t732(extent_1)
    real(c_double) :: t746(extent_1)
    real(c_double) :: t755(extent_1)
    real(c_double) :: t764(extent_1)
    real(c_double) :: t836(extent_1)
    real(c_double) :: t845(extent_1)
    real(c_double) :: t854(extent_1)
    real(c_double) :: t1058(extent_1)
    real(c_double) :: t1067(extent_1)
    real(c_double) :: t1076(extent_1)
    real(c_double) :: t1085(extent_1)
    real(c_double) :: t1094(extent_1)
    real(c_double) :: t1103(extent_1)
    real(c_double) :: t749(extent_16)
    real(c_double) :: t767(extent_16)
    real(c_double) :: t758(extent_16)
    real(c_double) :: t708(extent_16)
    real(c_double) :: t859(extent_1)
    real(c_double) :: t860(extent_16)
    real(c_double) :: t863(extent_16)
    real(c_double) :: t866(extent_16)
    real(c_double) :: t740(extent_16)
    real(c_double) :: t1387(extent_16)
    real(c_double) :: t793(extent_1)
    real(c_double) :: t794(extent_1)
    real(c_double) :: t800(extent_1)
    real(c_double) :: t806(extent_1)
    real(c_double) :: t957(extent_1)
    real(c_double) :: t1009(extent_1)
    real(c_double) :: t1025(extent_1)
    real(c_double) :: t967(extent_1)
    real(c_double) :: t1155(extent_16)
    real(c_double) :: t1162(extent_16)
    real(c_double) :: t1169(extent_16)
    real(c_double) :: t1176(extent_16)
    real(c_double) :: t1183(extent_16)
    real(c_double) :: t1190(extent_16)
    real(c_double) :: t1389(extent_16)
    real(c_double) :: t796(extent_1)
    real(c_double) :: t797(extent_16)
    real(c_double) :: t803(extent_16)
    real(c_double) :: t809(extent_16)
    real(c_double) :: t1299(extent_16)
    real(c_double) :: t1301(extent_1)
    real(c_double) :: t1305(extent_1)
    real(c_double) :: t1309(extent_1)
    real(c_double) :: t1300(extent_1)
    real(c_double) :: t1304(extent_1)
    real(c_double) :: t1308(extent_1)
    real(c_double) :: t1236(extent_16)
    real(c_double) :: t1238(extent_1)
    real(c_double) :: t1239(extent_16)
    real(c_double) :: t1256(extent_16)
    real(c_double) :: t1028(extent_1)
    real(c_double) :: t978(extent_1)
    real(c_double) :: t1267(extent_1)
    real(c_double) :: t1268(extent_16)
    real(c_double) :: t1270(extent_16)
    real(c_double) :: t1272(extent_1)
    real(c_double) :: t1276(extent_1)
    real(c_double) :: t1280(extent_1)
    real(c_double) :: t996(extent_16)
    real(c_double) :: t987(extent_16)
    real(c_double) :: t1006(extent_1)
    real(c_double) :: t1022(extent_1)
    real(c_double) :: t1041(extent_1)
    real(c_double) :: t1003(extent_1)
    real(c_double) :: t1019(extent_1)
    real(c_double) :: t1038(extent_1)
    real(c_double) :: t1002(extent_1)
    real(c_double) :: t1018(extent_1)
    real(c_double) :: t1037(extent_1)

    ! block entry
    t35 = (t0 + t1)
    t36 = (t2 - t3)
    t37 = 2.0_c_double
    t50 = 0.61_c_double
    t52 = 0.83_c_double
    t55 = 0.37_c_double
    t57 = 0.29_c_double
    t72 = (t7 + t8)
    t73 = (t9 + t10)
    t74 = (t11 + t12)
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t105 = 12.9898_c_double
    t107 = 78.233_c_double
    t115 = 39.3467_c_double
    t117 = 11.135_c_double
    t135 = -0.003_c_double
    t140 = 0.032_c_double
    t150 = (t5 - t17)
    t151 = (t6 - t18)
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t206 = (t18 - t22)
    t203 = (t17 - t19)
    t204 = (t18 - t20)
    t205 = (t17 - t21)
    t99 = 5.0_c_double
    t217 = (t99(1) - t17)
    t100 = 3.45_c_double
    t218 = (t100(1) - t18)
    t99 = 5.0_c_double
    t255 = 0.54_c_double
    t283 = 3.5_c_double
    t255 = 0.54_c_double
    t310 = (t5 - t19)
    t311 = (t6 - t20)
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t366 = (t19 - t21)
    t367 = (t20 - t22)
    t99 = 5.0_c_double
    t373 = (t99(1) - t19)
    t100 = 3.45_c_double
    t374 = (t100(1) - t20)
    t99 = 5.0_c_double
    t255 = 0.54_c_double
    t283 = 3.5_c_double
    t255 = 0.54_c_double
    t466 = (t5 - t21)
    t467 = (t6 - t22)
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t99 = 5.0_c_double
    t519 = (t99(1) - t21)
    t100 = 3.45_c_double
    t520 = (t100(1) - t22)
    t99 = 5.0_c_double
    t255 = 0.54_c_double
    t283 = 3.5_c_double
    t255 = 0.54_c_double
    t612 = 0.42_c_double
    t614 = 0.34_c_double
    t181 = 0.18_c_double
    t620 = 0.35_c_double
    t900 = 1.15_c_double
    t901 = (t5 - t900(1))
    t902 = 5.75_c_double
    t903 = (t6 - t902(1))
    t904 = 8.85_c_double
    t905 = (t5 - t904(1))
    t935 = 0.1_c_double
    t76 = 0.0_c_double
    t970 = 4.6_c_double
    t1136 = -0.095_c_double
    t1141 = -0.072_c_double
    t1145 = 3.2_c_double
    t1146 = (t1 * t1145(1))
    t1147 = 2.2_c_double
    t1148 = (t1 * t1147(1))
    t630 = 0.12_c_double
    t630 = 0.12_c_double
    t174 = 0.3_c_double
    t99 = 5.0_c_double
    t1283 = (t5 - t99(1))
    t100 = 3.45_c_double
    t1284 = (t6 - t100(1))
    t79 = 1.0_c_double
    t80 = min(max(t13, t76(1)), t79(1))
    t79 = 1.0_c_double
    t86 = min(max(t14, t76(1)), t79(1))
    t79 = 1.0_c_double
    t92 = min(max(t15, t76(1)), t79(1))
    t101 = 0.08_c_double
    t174 = 0.3_c_double
    t234 = 1.71_c_double
    t174 = 0.3_c_double
    t174 = 0.3_c_double
    t101 = 0.08_c_double
    t266 = 1.37_c_double
    t174 = 0.3_c_double
    t174 = 0.3_c_double
    t101 = 0.08_c_double
    t174 = 0.3_c_double
    t390 = 1.63_c_double
    t174 = 0.3_c_double
    t174 = 0.3_c_double
    t101 = 0.08_c_double
    t422 = 1.43_c_double
    t174 = 0.3_c_double
    t174 = 0.3_c_double
    t101 = 0.08_c_double
    t174 = 0.3_c_double
    t536 = 1.79_c_double
    t174 = 0.3_c_double
    t174 = 0.3_c_double
    t101 = 0.08_c_double
    t568 = 1.31_c_double
    t174 = 0.3_c_double
    t174 = 0.3_c_double
    t101 = 0.08_c_double
    t612 = 0.42_c_double
    t955 = min(max(t31, t76(1)), t612(1))
    t1109 = 2.11_c_double
    t1117 = 1.91_c_double
    t1127 = 2.27_c_double
    t1139 = exp((t1 * t1136(1)))
    t1144 = exp((t1 * t1141(1)))
    t40 = (((t4 * t37(1)) - t2) - t3)
    t79 = 1.0_c_double
    t94 = (t79(1) - t80)
    t79 = 1.0_c_double
    t96 = (t79(1) - t86)
    t79 = 1.0_c_double
    t98 = (t79(1) - t92)
    t104 = real(floor((t35 * t101(1))), c_double)
    t154 = ((t150 * t150) + (t151 * t151))
    t228 = 1.85_c_double
    t236 = 0.2_c_double
    t228 = 1.85_c_double
    t268 = 1.1_c_double
    t314 = ((t310 * t310) + (t311 * t311))
    t228 = 1.85_c_double
    t392 = 2.3_c_double
    t228 = 1.85_c_double
    t424 = 2.8_c_double
    t470 = ((t466 * t466) + (t467 * t467))
    t228 = 1.85_c_double
    t538 = 4.2_c_double
    t228 = 1.85_c_double
    t99 = 5.0_c_double
    t908 = ((t901 * t901) + (t903 * t903))
    t911 = ((t905 * t905) + (t903 * t903))
    t963 = 0.16_c_double
    t1119 = 2.1_c_double
    t538 = 4.2_c_double
    t62 = 0.72_c_double
    t110 = 37.719_c_double
    t120 = 19.913_c_double
    t101 = 0.08_c_double
    t168 = 0.14_c_double
    t171 = 1.2_c_double
    t177 = 1.65_c_double
    t181 = 0.18_c_double
    t211 = (((t203 * t203) + (t204 * t204)) + t181(1))
    t181 = 0.18_c_double
    t216 = (((t205 * t205) + (t206 * t206)) + t181(1))
    t101 = 0.08_c_double
    t101 = 0.08_c_double
    t168 = 0.14_c_double
    t171 = 1.2_c_double
    t177 = 1.65_c_double
    t181 = 0.18_c_double
    t372 = (((t366 * t366) + (t367 * t367)) + t181(1))
    t101 = 0.08_c_double
    t101 = 0.08_c_double
    t168 = 0.14_c_double
    t171 = 1.2_c_double
    t177 = 1.65_c_double
    t101 = 0.08_c_double
    t959 = 0.22_c_double
    t64 = (((t5 * t50(1)) + (t6 * t52(1))) + (sin(((t5 * t55(1)) - (t6 * t57(1)))) * t62(1)))
    t164 = 4.805000000000001_c_double
    t225 = sqrt((((t217 * t217) + (t218 * t218)) + t101(1)))
    t233 = 0.58_c_double
    t233 = 0.58_c_double
    t324 = 4.805000000000001_c_double
    t381 = sqrt((((t373 * t373) + (t374 * t374)) + t101(1)))
    t233 = 0.58_c_double
    t233 = 0.58_c_double
    t480 = 4.805000000000001_c_double
    t527 = sqrt((((t519 * t519) + (t520 * t520)) + t101(1)))
    t233 = 0.58_c_double
    t233 = 0.58_c_double
    t76 = 0.0_c_double
    t917 = 0.4608_c_double
    t926 = 0.6728_c_double
    t711 = 0.78_c_double
    t1114 = ((sin((t35 * t1109(1))) * t959(1)) + t711(1))
    t959 = 0.22_c_double
    t959 = 0.22_c_double
    t1293 = 0.23120000000000004_c_double
    t1346 = 0.1058_c_double
    t1355 = 0.1058_c_double
    t44 = 1.0e-05_c_double
    t66 = cos(t64)
    t68 = sin(t64)
    t79 = 1.0_c_double
    t920 = exp(((-t908) / t917(1)))
    t929 = exp(((-t911) / t926(1)))
    t711 = 0.78_c_double
    t1124 = ((sin(((t35 * t1117(1)) + t1119(1))) * t959(1)) + t711(1))
    t711 = 0.78_c_double
    t1134 = ((sin(((t35 * t1127(1)) + t538(1))) * t959(1)) + t711(1))
    t1296 = exp(((-((t1283 * t1283) + (t1284 * t1284))) / t1293(1)))
    t1349 = exp(((-t908) / t1346(1)))
    t330 = (exp(((-t314) / t324(1))) / (t314 + t168(1)))
    t1358 = exp(((-t911) / t1355(1)))
    t47 = sqrt((((t36 * t36) + (t40 * t40)) + t44(1)))
    t170 = (exp(((-t154) / t164(1))) / (t154 + t168(1)))
    t486 = (exp(((-t470) / t480(1))) / (t470 + t168(1)))
    t630 = 0.12_c_double
    t79 = 1.0_c_double
    t1313 = (t79(1) - t1296)
    t79 = 1.0_c_double
    t1360 = (t79(1) - t1349)
    t79 = 1.0_c_double
    t1362 = (t79(1) - t1358)
    t1364 = 53.0_c_double
    t1368 = 83.0_c_double
    t1372 = 103.0_c_double
    t1376 = 205.0_c_double
    t1380 = 245.0_c_double
    t1384 = 252.0_c_double
    t49 = (t40 / t47)
    t48 = (t36 / t47)
    t62 = 0.72_c_double
    t181 = 0.18_c_double
    t632 = ((min(max(((((t2 * t612(1)) + (t4 * t614(1))) + (t3 * t181(1))) + (t29 * t620(1))), t76(1)), t79(1)) * t630(1)) + t181(1))
    t79 = 1.0_c_double
    t101 = 0.08_c_double
    t187 = 1.0e-06_c_double
    t1325 = 0.6_c_double
    t1326 = min((t929 * t955), t1325(1))
    t76 = 0.0_c_double
    t187 = 1.0e-06_c_double
    t187 = 1.0e-06_c_double
    t187 = 1.0e-06_c_double
    t79 = 1.0_c_double
    t1214 = (t79(1) - min(((t1 * t630(1)) * t920), t101(1)))
    t79 = 1.0_c_double
    t1328 = (t79(1) - t1326)
    t1330 = 218.0_c_double
    t1334 = 249.0_c_double
    t1016 = 255.0_c_double
    t71 = ((t66 * t48) + (t68 * t49))
    t131 = 0.28_c_double
    t132 = (max(((sin((((t5 * t105(1)) + (t6 * t107(1))) + (t104 * t110(1)))) * cos((((t5 * t115(1)) - (t6 * t117(1))) + (t104 * t120(1))))) - t62(1)), t76(1)) / t131(1))
    t181 = 0.18_c_double
    t181 = 0.18_c_double
    t181 = 0.18_c_double
    t76 = 0.0_c_double
    t79 = 1.0_c_double
    t149 = min(max(((t16 * exp((t1 * t135(1)))) + ((t1 * t140(1)) * (t132 * t132))), t76(1)), t79(1))
    t190 = ((t170 * (((((t94 * t171(1)) * t74) - (t72 * t174(1))) + ((t94 * t177(1)) * t149)) + (t71 * t181(1)))) / (sqrt((t154 + t101(1))) * (sum(t170) + t187(1))))
    t353 = ((t330 * (((((t96 * t171(1)) * t72) - (t73 * t174(1))) + ((t96 * t177(1)) * t149)) + (((t68 * t48) - (t66 * t49)) * t181(1)))) / (sqrt((t314 + t101(1))) * (sum(t330) + t187(1))))
    t506 = ((t486 * (((((t98 * t171(1)) * t73) - (t74 * t174(1))) + ((t98 * t177(1)) * t149)) - (t71 * t181(1)))) / (sqrt((t470 + t101(1))) * (sum(t486) + t187(1))))
    t226 = 2.35_c_double
    t226 = 2.35_c_double
    t226 = 2.35_c_double
    t226 = 2.35_c_double
    t226 = 2.35_c_double
    t226 = 2.35_c_double
    t291 = (t23 + ((((((((((t5 * t76(1)) + sum((t150 * t190))) * t226(1)) + (((t80 * t228(1)) * t217) / t225)) + (cos(((t35 * t234(1)) + t236(1))) * t233(1))) + ((t203 * t174(1)) / t211)) + ((t205 * t174(1)) / t216)) + ((t99(1) - t17) * t101(1))) - (t23 * t255(1))) * t1))
    t293 = (t24 + ((((((((((t6 * t76(1)) + sum((t151 * t190))) * t226(1)) + (((t80 * t228(1)) * t218) / t225)) + (sin(((t35 * t266(1)) + t268(1))) * t233(1))) + ((t204 * t174(1)) / t211)) + ((t206 * t174(1)) / t216)) + ((t283(1) - t18) * t101(1))) - (t24 * t255(1))) * t1))
    t447 = (t25 + ((((((((((t5 * t76(1)) + sum((t310 * t353))) * t226(1)) + (((t86 * t228(1)) * t373) / t381)) + (cos(((t35 * t390(1)) + t392(1))) * t233(1))) - ((t203 * t174(1)) / t211)) + ((t366 * t174(1)) / t372)) + ((t99(1) - t19) * t101(1))) - (t25 * t255(1))) * t1))
    t449 = (t26 + ((((((((((t6 * t76(1)) + sum((t311 * t353))) * t226(1)) + (((t86 * t228(1)) * t374) / t381)) + (sin(((t35 * t422(1)) + t424(1))) * t233(1))) - ((t204 * t174(1)) / t211)) + ((t367 * t174(1)) / t372)) + ((t283(1) - t20) * t101(1))) - (t26 * t255(1))) * t1))
    t593 = (t27 + ((((((((((t5 * t76(1)) + sum((t466 * t506))) * t226(1)) + (((t92 * t228(1)) * t519) / t527)) + (cos(((t35 * t536(1)) + t538(1))) * t233(1))) - ((t205 * t174(1)) / t216)) - ((t366 * t174(1)) / t372)) + ((t99(1) - t21) * t101(1))) - (t27 * t255(1))) * t1))
    t595 = (t28 + ((((((((((t6 * t76(1)) + sum((t467 * t506))) * t226(1)) + (((t92 * t228(1)) * t520) / t527)) + (sin(((t35 * t568(1)) + t99(1))) * t233(1))) - ((t206 * t174(1)) / t216)) - ((t367 * t174(1)) / t372)) + ((t283(1) - t22) * t101(1))) - (t28 * t255(1))) * t1))
    t297 = 0.65_c_double
    t297 = 0.65_c_double
    t297 = 0.65_c_double
    t297 = 0.65_c_double
    t297 = 0.65_c_double
    t297 = 0.65_c_double
    t300 = 9.35_c_double
    t301 = min(max((t17 + (t291 * t1)), t297(1)), t300(1))
    t308 = 6.35_c_double
    t309 = min(max((t18 + (t293 * t1)), t297(1)), t308(1))
    t300 = 9.35_c_double
    t457 = min(max((t19 + (t447 * t1)), t297(1)), t300(1))
    t308 = 6.35_c_double
    t465 = min(max((t20 + (t449 * t1)), t297(1)), t308(1))
    t300 = 9.35_c_double
    t603 = min(max((t21 + (t593 * t1)), t297(1)), t300(1))
    t308 = 6.35_c_double
    t611 = min(max((t22 + (t595 * t1)), t297(1)), t308(1))
    t99 = 5.0_c_double
    t99 = 5.0_c_double
    t100 = 3.45_c_double
    t100 = 3.45_c_double
    t99 = 5.0_c_double
    t99 = 5.0_c_double
    t100 = 3.45_c_double
    t100 = 3.45_c_double
    t99 = 5.0_c_double
    t99 = 5.0_c_double
    t100 = 3.45_c_double
    t100 = 3.45_c_double
    t646 = (((t5 - t457) * (t5 - t457)) + ((t6 - t465) * (t6 - t465)))
    t639 = (((t5 - t301) * (t5 - t301)) + ((t6 - t309) * (t6 - t309)))
    t653 = (((t5 - t603) * (t5 - t603)) + ((t6 - t611) * (t6 - t611)))
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t714 = 1.2168_c_double
    t723 = 1.2168_c_double
    t732 = 1.2168_c_double
    t746 = 0.18_c_double
    t755 = 0.18_c_double
    t764 = 0.18_c_double
    t836 = 0.32000000000000006_c_double
    t845 = 0.32000000000000006_c_double
    t854 = 0.32000000000000006_c_double
    t1058 = 0.1152_c_double
    t1067 = 0.2888_c_double
    t1076 = 0.1152_c_double
    t1085 = 0.2888_c_double
    t1094 = 0.1152_c_double
    t1103 = 0.2888_c_double
    t749 = exp(((-t639) / t746(1)))
    t767 = exp(((-t653) / t764(1)))
    t758 = exp(((-t646) / t755(1)))
    t708 = max(max((min(max((t632 - abs((t5 - t301))), t76(1)), max((t632 - abs((t6 - t309))), t76(1))) / t632), (min(max((t632 - abs((t5 - t457))), t76(1)), max((t632 - abs((t6 - t465))), t76(1))) / t632)), (min(max((t632 - abs((t5 - t603))), t76(1)), max((t632 - abs((t6 - t611))), t76(1))) / t632))
    t187 = 1.0e-06_c_double
    t187 = 1.0e-06_c_double
    t187 = 1.0e-06_c_double
    t859 = 2.4_c_double
    t860 = ((t80 * exp(((-(((t301 - t99(1)) * (t301 - t99(1))) + ((t309 - t100(1)) * (t309 - t100(1))))) / t836(1)))) * t859(1))
    t859 = 2.4_c_double
    t863 = ((t86 * exp(((-(((t457 - t99(1)) * (t457 - t99(1))) + ((t465 - t100(1)) * (t465 - t100(1))))) / t845(1)))) * t859(1))
    t859 = 2.4_c_double
    t866 = ((t92 * exp(((-(((t603 - t99(1)) * (t603 - t99(1))) + ((t611 - t100(1)) * (t611 - t100(1))))) / t854(1)))) * t859(1))
    t79 = 1.0_c_double
    t740 = min(((exp(((-t639) / t714(1))) + exp(((-t646) / t723(1)))) + exp(((-t653) / t732(1)))), t79(1))
    t959 = 0.22_c_double
    t1387 = (t708 * t708)
    t793 = 0.7_c_double
    t794 = min((sum((t149 * t749)) / (sum(t749) + t187(1))), t793)
    t793 = 0.7_c_double
    t800 = min((sum((t149 * t758)) / (sum(t758) + t187(1))), t793)
    t793 = 0.7_c_double
    t806 = min((sum((t149 * t767)) / (sum(t767) + t187(1))), t793)
    t957 = -0.42_c_double
    t1009 = 54.0_c_double
    t1025 = 30.0_c_double
    t967 = 20.0_c_double
    t79 = 1.0_c_double
    t1155 = min(((t7 * t1139) + ((t1146 * exp(((-t639) / t1058(1)))) * t1114)), t79(1))
    t79 = 1.0_c_double
    t1162 = min(((t8 * t1144) + ((t1148 * exp(((-t639) / t1067(1)))) * t1114)), t79(1))
    t79 = 1.0_c_double
    t1169 = min(((t9 * t1139) + ((t1146 * exp(((-t646) / t1076(1)))) * t1124)), t79(1))
    t79 = 1.0_c_double
    t1176 = min(((t10 * t1144) + ((t1148 * exp(((-t646) / t1085(1)))) * t1124)), t79(1))
    t79 = 1.0_c_double
    t1183 = min(((t11 * t1139) + ((t1146 * exp(((-t653) / t1094(1)))) * t1134)), t79(1))
    t79 = 1.0_c_double
    t1190 = min(((t12 * t1144) + ((t1148 * exp(((-t653) / t1103(1)))) * t1134)), t79(1))
    t79 = 1.0_c_double
    t1389 = (t79(1) - t1387)
    t1380 = 245.0_c_double
    t1384 = 252.0_c_double
    t1016 = 255.0_c_double
    t1219 = (t1183 * t1214)
    t1220 = (t1190 * t1214)
    t1218 = (t1176 * t1214)
    t1216 = (t1162 * t1214)
    t1217 = (t1169 * t1214)
    t1215 = (t1155 * t1214)
    t796 = 1.35_c_double
    t797 = ((t94 * t794(1)) * t796(1))
    t796 = 1.35_c_double
    t803 = ((t96 * t800(1)) * t796(1))
    t796 = 1.35_c_double
    t809 = ((t98 * t806(1)) * t796(1))
    t76 = 0.0_c_double
    t949 = max((t30 + (t1 * ((t860 + t863) + t866))), t76(1))
    t967 = 20.0_c_double
    t1016 = 255.0_c_double
    t1016 = 255.0_c_double
    t1016 = 255.0_c_double
    t79 = 1.0_c_double
    t1299 = min(t949, t79(1))
    t1301 = 58.0_c_double
    t1305 = 42.0_c_double
    t1309 = 24.0_c_double
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t1300 = 102.0_c_double
    t1304 = 72.0_c_double
    t1308 = 48.0_c_double
    t79 = 1.0_c_double
    t875 = min(max((t80 + (t1 * (t797 - t860))), t76(1)), t79(1))
    t79 = 1.0_c_double
    t884 = min(max((t86 + (t1 * (t803 - t863))), t76(1)), t79(1))
    t79 = 1.0_c_double
    t893 = min(max((t92 + (t1 * (t809 - t866))), t76(1)), t79(1))
    t974 = (t33 + (((((((t740 * t957(1)) - ((t708 * t959(1)) * t708)) + ((t955 * t963(1)) * t929)) - t32) * t967(1)) - (t33 * t970(1))) * t1))
    t187 = 1.0e-06_c_double
    t1236 = max((((((t1215 + t1216) + t1217) + t1218) + t1219) + t1220), t187(1))
    t76 = 0.0_c_double
    t1238 = 0.88_c_double
    t1239 = min(t1236, t1238(1))
    t976 = (t32 + (t974 * t1))
    t79 = 1.0_c_double
    t79 = 1.0_c_double
    t1256 = (t79(1) - t1239)
    t79 = 1.0_c_double
    t942 = min((max((t149 - (t1 * (((t749 * t797) + (t758 * t803)) + (t767 * t809)))), t76(1)) * (t79(1) - ((t1 * t935(1)) * t920))), t79(1))
    t1028 = 8.0_c_double
    t978 = 0.5_c_double
    t612 = 0.42_c_double
    t1267 = 0.76_c_double
    t1268 = min(t942, t1267(1))
    t99 = 5.0_c_double
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t1228 = max(((t31 + (((t1 * t630(1)) * sum(((((((t1155 + t1162) + t1169) + t1176) + t1183) + t1190) * t920))) / (sum(t920) + t187(1)))) - ((t1 * t174(1)) * t955)), t76(1))
    t79 = 1.0_c_double
    t1270 = (t79(1) - t1268)
    t1272 = 226.0_c_double
    t1276 = 181.0_c_double
    t1280 = 62.0_c_double
    t76 = 0.0_c_double
    t79 = 1.0_c_double
    t996 = min(max(((-t976) / t612(1)), t76(1)), t79(1))
    t79 = 1.0_c_double
    t987 = min(max((((t34 + t976) - t978(1)) / t99(1)), t76(1)), t79(1))
    t1006 = 34.0_c_double
    t1022 = 21.0_c_double
    t1041 = 15.0_c_double
    t1003 = 27.0_c_double
    t1019 = 18.0_c_double
    t1038 = 16.0_c_double
    t1002 = 186.0_c_double
    t1018 = 220.0_c_double
    t1037 = 232.0_c_double
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t1016 = 255.0_c_double
    t76 = 0.0_c_double
    t1016 = 255.0_c_double
    t1016 = 255.0_c_double
    t1397 = ((((((((((((((min(max((((((t987 * t1019(1)) + t1018(1)) - (t996 * t1022(1))) + (t740 * t1025(1))) + (min(abs(t974), t79(1)) * t1028(1))), t76(1)), t1016(1)) * t1256) + (((((t1216 + t1217) + t1218) * t1016(1)) / t1236) * t1239)) * t1270) + (t1268 * t1276(1))) * t1313) + (((t1299 * t1305(1)) + t1304(1)) * t1296)) * t1328) + (t1326 * t1334(1))) * t1360) + (t1349 * t1368(1))) * t1362) + (t1358 * t1380(1))) * t1389) + (t1387 * t1384(1)))
    t1401 = ((((((((((((((min(max(((((t987 * t1038(1)) + t1037(1)) + (t996 * t1041(1))) + (t740 * t967(1))), t76(1)), t1016(1)) * t1256) + (((((t1218 + t1219) + t1220) * t1016(1)) / t1236) * t1239)) * t1270) + (t1268 * t1280(1))) * t1313) + (((t1299 * t1309(1)) + t1308(1)) * t1296)) * t1328) + (t1326 * t1016(1))) * t1360) + (t1349 * t1372(1))) * t1362) + (t1358 * t1384(1))) * t1389) + (t1387 * t1016(1)))
    t1393 = ((((((((((((((min(max(((((t987 * t1003(1)) + t1002(1)) - (t996 * t1006(1))) + (t740 * t1009(1))), t76(1)), t1016(1)) * t1256) + (((((t1215 + t1216) + t1220) * t1016(1)) / t1236) * t1239)) * t1270) + (t1268 * t1272(1))) * t1313) + (((t1299 * t1301(1)) + t1300(1)) * t1296)) * t1328) + (t1326 * t1330(1))) * t1360) + (t1349 * t1364(1))) * t1362) + (t1358 * t1376(1))) * t1389) + (t1387 * t1380(1)))
    return
  end subroutine numerical_region_0

end module kernel_fortran
