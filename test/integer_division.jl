
@test @inferred(
        Octavian.cld_fast(Octavian.StaticInt(6), Octavian.StaticInt(2))
      ) ==
      @inferred(Octavian.cld_fast(11, Octavian.StaticInt(4))) ==
      @inferred(Octavian.cld_fast(Octavian.StaticInt(7), 3)) ==
      @inferred(Octavian.cld_fast(9, 4)) ==
      3
@test @inferred(
        Octavian.cld_fast(Octavian.StaticInt(7), Octavian.StaticInt(2))
      ) ==
      @inferred(Octavian.cld_fast(13, Octavian.StaticInt(4))) ==
      @inferred(Octavian.cld_fast(Octavian.StaticInt(8), 2)) ==
      @inferred(Octavian.cld_fast(15, 4)) ==
      4
