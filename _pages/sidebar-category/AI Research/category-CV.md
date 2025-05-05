---
title: "CV"
layout: archive
permalink: /CV/
---
{% assign posts = site.categories.CV %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}