# Copyright: (C) 2022 Istituto Italiano di Tecnologia | Robotics, Brains and Cognitive Science
# Authors: Luca Garello
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

set(appname ${PROJECT_NAME})

file(GLOB poses ${CMAKE_CURRENT_SOURCE_DIR}/conf/*.sh)
file(GLOB conf ${CMAKE_CURRENT_SOURCE_DIR}/conf/*.ini)
file(GLOB scripts ${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.xml)

yarp_install(PROGRAMS ${poses} DESTINATION ${ICUBCONTRIB_CONTEXTS_INSTALL_DIR}/${appname})
yarp_install(FILES ${conf} DESTINATION ${ICUBCONTRIB_CONTEXTS_INSTALL_DIR}/${appname})
yarp_install(FILES ${scripts} DESTINATION ${ICUBCONTRIB_APPLICATIONS_INSTALL_DIR})


