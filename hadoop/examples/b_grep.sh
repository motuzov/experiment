#!/bin/sh
hadoop com.sun.tools.javac.Main GrepSessions.java 
jar cf grep.jar GrepSessions*.class 
