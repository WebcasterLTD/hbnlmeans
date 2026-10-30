# libhbnlmeans
#   make                     build static and shared libraries
#   make install             install to PREFIX (default /usr/local)
#   make install SHARED=0    static library only
#   make check               build and run the self test

VERSION = 1.0.0
SOVER   = 1

PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib
INCDIR ?= $(PREFIX)/include
PCDIR  ?= $(LIBDIR)/pkgconfig
SHARED ?= 1

CC     ?= cc
AR     ?= ar
CFLAGS ?= -O3
CFLAGS += -std=gnu11 -fPIC -Wall -Wextra -Wno-unused-parameter -Iinclude

LIBS = libhbnlmeans.a
ifeq ($(SHARED),1)
LIBS += libhbnlmeans.so.$(VERSION)
endif

all: $(LIBS) hbnlmeans.pc

src/hbnlmeans.o: src/hbnlmeans.c src/hbnlmeans_template.c include/hbnlmeans.h
	$(CC) $(CFLAGS) -c -o $@ src/hbnlmeans.c

libhbnlmeans.a: src/hbnlmeans.o
	$(AR) rcs $@ $^

libhbnlmeans.so.$(VERSION): src/hbnlmeans.o
	$(CC) -shared -Wl,-soname,libhbnlmeans.so.$(SOVER) -o $@ $^ -lm
	ln -sf $@ libhbnlmeans.so.$(SOVER)
	ln -sf $@ libhbnlmeans.so

hbnlmeans.pc: hbnlmeans.pc.in Makefile
	sed -e 's|@PREFIX@|$(PREFIX)|' -e 's|@LIBDIR@|$(LIBDIR)|' \
	    -e 's|@INCDIR@|$(INCDIR)|' -e 's|@VERSION@|$(VERSION)|' $< > $@

test/selftest: test/selftest.c libhbnlmeans.a
	$(CC) $(CFLAGS) -o $@ $< libhbnlmeans.a -lm

check: test/selftest
	./test/selftest

install: all
	install -d $(DESTDIR)$(LIBDIR) $(DESTDIR)$(INCDIR) $(DESTDIR)$(PCDIR)
	install -m 644 include/hbnlmeans.h $(DESTDIR)$(INCDIR)/
	install -m 644 libhbnlmeans.a $(DESTDIR)$(LIBDIR)/
ifeq ($(SHARED),1)
	install -m 755 libhbnlmeans.so.$(VERSION) $(DESTDIR)$(LIBDIR)/
	ln -sf libhbnlmeans.so.$(VERSION) $(DESTDIR)$(LIBDIR)/libhbnlmeans.so.$(SOVER)
	ln -sf libhbnlmeans.so.$(VERSION) $(DESTDIR)$(LIBDIR)/libhbnlmeans.so
endif
	install -m 644 hbnlmeans.pc $(DESTDIR)$(PCDIR)/

uninstall:
	rm -f $(DESTDIR)$(INCDIR)/hbnlmeans.h $(DESTDIR)$(LIBDIR)/libhbnlmeans.* $(DESTDIR)$(PCDIR)/hbnlmeans.pc

clean:
	rm -f src/*.o libhbnlmeans.a libhbnlmeans.so* hbnlmeans.pc test/selftest

.PHONY: all check install uninstall clean
