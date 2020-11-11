{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{
			"major" : 8,
			"minor" : 1,
			"revision" : 4,
			"architecture" : "x64",
			"modernui" : 1
		}
,
		"classnamespace" : "box",
		"rect" : [ 309.0, 93.0, 526.0, 800.0 ],
		"bglocked" : 0,
		"openinpresentation" : 1,
		"default_fontsize" : 12.0,
		"default_fontface" : 0,
		"default_fontname" : "Arial",
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"gridsnaponopen" : 1,
		"objectsnaponopen" : 1,
		"statusbarvisible" : 2,
		"toolbarvisible" : 1,
		"lefttoolbarpinned" : 0,
		"toptoolbarpinned" : 0,
		"righttoolbarpinned" : 0,
		"bottomtoolbarpinned" : 0,
		"toolbars_unpinned_last_save" : 0,
		"tallnewobj" : 0,
		"boxanimatetime" : 200,
		"enablehscroll" : 1,
		"enablevscroll" : 1,
		"devicewidth" : 0.0,
		"description" : "",
		"digest" : "",
		"tags" : "",
		"style" : "",
		"subpatcher_template" : "",
		"assistshowspatchername" : 0,
		"boxes" : [ 			{
				"box" : 				{
					"id" : "obj-11",
					"linecount" : 5,
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 165.0, 219.0, 50.0, 76.0 ],
					"text" : "FullPacket 28 105553157636928"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-6",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 164.0, 380.0, 50.0, 22.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-2",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"patching_rect" : [ 361.5, 322.0, 117.0, 22.0 ],
					"text" : "routepass interp_ms"
				}

			}
, 			{
				"box" : 				{
					"bgmode" : 0,
					"border" : 0,
					"clickthrough" : 0,
					"enablehscroll" : 0,
					"enablevscroll" : 0,
					"id" : "obj-3",
					"lockeddragscroll" : 0,
					"maxclass" : "bpatcher",
					"name" : "acids.model.drop.maxpat",
					"numinlets" : 1,
					"numoutlets" : 1,
					"offset" : [ 0.0, 0.0 ],
					"outlettype" : [ "" ],
					"patching_rect" : [ 324.5, 82.0, 128.0, 128.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 6.0, 132.0, 504.0, 230.0 ],
					"viewvisibility" : 1
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-66",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 121.0, 51.0, 156.0, 22.0 ],
					"text" : "port_in 1234, port_out 1235"
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"fontsize" : 20.0,
					"id" : "obj-58",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 355.0, 265.5, 244.0, 29.0 ],
					"text" : "Direct output"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-57",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 25.0, 300.0, 100.0, 22.0 ],
					"text" : "r #0_from_server"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-56",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 490.5, 322.0, 100.0, 22.0 ],
					"text" : "r #0_from_server"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-53",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 501.5, 511.5, 128.0, 22.0 ],
					"text" : "send~ #0_main_out_2"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-54",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 361.5, 511.5, 128.0, 22.0 ],
					"text" : "send~ #0_main_out_1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-51",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 25.0, 610.0, 88.0, 22.0 ],
					"text" : "s #0_to_server"
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"fontsize" : 20.0,
					"id" : "obj-50",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 25.0, 265.5, 244.0, 29.0 ],
					"text" : "Control"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-49",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 653.0, 227.0, 88.0, 22.0 ],
					"text" : "s #0_to_server"
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"fontsize" : 20.0,
					"id" : "obj-48",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 644.5, 14.5, 244.0, 29.0 ],
					"text" : "Direct input"
				}

			}
, 			{
				"box" : 				{
					"bgmode" : 0,
					"border" : 0,
					"clickthrough" : 0,
					"enablehscroll" : 0,
					"enablevscroll" : 0,
					"id" : "obj-47",
					"lockeddragscroll" : 0,
					"maxclass" : "bpatcher",
					"name" : "acids.fft.out.maxpat",
					"numinlets" : 1,
					"numoutlets" : 2,
					"offset" : [ 0.0, 0.0 ],
					"outlettype" : [ "signal", "signal" ],
					"patching_rect" : [ 361.5, 363.0, 159.0, 128.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 6.0, 611.0, 504.0, 126.0 ],
					"viewvisibility" : 1
				}

			}
, 			{
				"box" : 				{
					"bgmode" : 0,
					"border" : 0,
					"clickthrough" : 0,
					"enablehscroll" : 0,
					"enablevscroll" : 0,
					"id" : "obj-42",
					"lockeddragscroll" : 0,
					"maxclass" : "bpatcher",
					"name" : "acids.sliders.maxpat",
					"numinlets" : 1,
					"numoutlets" : 2,
					"offset" : [ 0.0, 0.0 ],
					"outlettype" : [ "", "" ],
					"patching_rect" : [ 25.0, 395.0, 128.0, 128.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 6.0, 361.0, 504.0, 248.0 ],
					"viewvisibility" : 1
				}

			}
, 			{
				"box" : 				{
					"bgmode" : 0,
					"border" : 0,
					"clickthrough" : 0,
					"enablehscroll" : 0,
					"enablevscroll" : 0,
					"id" : "obj-35",
					"lockeddragscroll" : 0,
					"maxclass" : "bpatcher",
					"name" : "acids.audio.in.maxpat",
					"numinlets" : 0,
					"numoutlets" : 1,
					"offset" : [ 0.0, 0.0 ],
					"outlettype" : [ "" ],
					"patching_rect" : [ 653.0, 78.0, 287.0, 122.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 162.0, 5.0, 348.0, 123.0 ],
					"viewvisibility" : 1
				}

			}
, 			{
				"box" : 				{
					"bgmode" : 0,
					"border" : 0,
					"clickthrough" : 0,
					"enablehscroll" : 0,
					"enablevscroll" : 0,
					"id" : "obj-1",
					"lockeddragscroll" : 0,
					"maxclass" : "bpatcher",
					"name" : "acids.player.maxpat",
					"numinlets" : 2,
					"numoutlets" : 0,
					"offset" : [ 0.0, 0.0 ],
					"patching_rect" : [ 726.0, 381.0, 154.0, 53.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 6.0, 77.0, 154.0, 53.0 ],
					"viewvisibility" : 1
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"fontsize" : 20.0,
					"id" : "obj-24",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 726.0, 309.0, 117.0, 29.0 ],
					"text" : "Direct audio"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-20",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 324.5, 51.0, 100.0, 22.0 ],
					"text" : "r #0_from_server"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-13",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 882.0, 345.5, 142.0, 22.0 ],
					"text" : "receive~ #0_main_out_2"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-15",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 726.0, 345.5, 142.0, 22.0 ],
					"text" : "receive~ #0_main_out_1"
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"fontsize" : 20.0,
					"id" : "obj-23",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 16.0, 14.5, 69.0, 29.0 ],
					"text" : "Server"
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"fontsize" : 20.0,
					"id" : "obj-22",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 313.5, 14.5, 244.0, 29.0 ],
					"text" : "Loading models"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-12",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 324.5, 227.0, 88.0, 22.0 ],
					"text" : "s #0_to_server"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-10",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 16.0, 166.0, 104.0, 22.0 ],
					"text" : "s #0_from_server"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-9",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 16.0, 51.0, 88.0, 22.0 ],
					"text" : "r #0_to_server"
				}

			}
, 			{
				"box" : 				{
					"args" : [ 0 ],
					"bgmode" : 0,
					"border" : 0,
					"clickthrough" : 0,
					"enablehscroll" : 0,
					"enablevscroll" : 0,
					"id" : "obj-5",
					"lockeddragscroll" : 0,
					"maxclass" : "bpatcher",
					"name" : "acids.osc.live.maxpat",
					"numinlets" : 1,
					"numoutlets" : 1,
					"offset" : [ 0.0, 0.0 ],
					"outlettype" : [ "" ],
					"patching_rect" : [ 16.0, 82.0, 151.0, 68.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 6.0, 5.0, 151.0, 68.0 ],
					"varname" : "acids.osc.live",
					"viewvisibility" : 1
				}

			}
 ],
		"lines" : [ 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 1 ],
					"source" : [ "obj-13", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"source" : [ "obj-15", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-47", 0 ],
					"source" : [ "obj-2", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-3", 0 ],
					"source" : [ "obj-20", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-12", 0 ],
					"source" : [ "obj-3", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 0 ],
					"source" : [ "obj-35", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-2", 0 ],
					"source" : [ "obj-42", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-51", 0 ],
					"source" : [ "obj-42", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-53", 0 ],
					"source" : [ "obj-47", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-54", 0 ],
					"source" : [ "obj-47", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 0 ],
					"order" : 1,
					"source" : [ "obj-5", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-11", 1 ],
					"order" : 0,
					"source" : [ "obj-5", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-47", 0 ],
					"source" : [ "obj-56", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-42", 0 ],
					"source" : [ "obj-57", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5", 0 ],
					"source" : [ "obj-66", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5", 0 ],
					"source" : [ "obj-9", 0 ]
				}

			}
 ],
		"parameters" : 		{
			"obj-47::obj-44" : [ "live.gain~[3]", "live.gain~[1]", 0 ],
			"obj-35::obj-5" : [ "live.gain~", "live.gain~", 0 ],
			"obj-35::obj-44" : [ "live.gain~[2]", "live.gain~[1]", 0 ],
			"obj-1::obj-82" : [ "Volume[1]", "Volume", 0 ],
			"parameterbanks" : 			{

			}
,
			"parameter_overrides" : 			{
				"obj-35::obj-5" : 				{
					"parameter_longname" : "live.gain~",
					"parameter_shortname" : "live.gain~"
				}
,
				"obj-1::obj-82" : 				{
					"parameter_longname" : "Volume[1]"
				}

			}
,
			"inherited_shortname" : 1
		}
,
		"dependency_cache" : [ 			{
				"name" : "acids.osc.live.maxpat",
				"bootpath" : "~/Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"patcherrelativepath" : "../../../../Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "acids.button.graphics.maxpat",
				"bootpath" : "~/Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"patcherrelativepath" : "../../../../Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "internal.png",
				"bootpath" : "~/Dropbox/code/acids/team/axel/vschaos_package/max/plugin/graphics/icons_white_50",
				"patcherrelativepath" : "../../../../Dropbox/code/acids/team/axel/vschaos_package/max/plugin/graphics/icons_white_50",
				"type" : "PNG",
				"implicit" : 1
			}
, 			{
				"name" : "external.png",
				"bootpath" : "~/Dropbox/code/acids/team/axel/vschaos_package/max/plugin/graphics/icons_white_50",
				"patcherrelativepath" : "../../../../Dropbox/code/acids/team/axel/vschaos_package/max/plugin/graphics/icons_white_50",
				"type" : "PNG",
				"implicit" : 1
			}
, 			{
				"name" : "ACIDS_logo_ctrl.png",
				"bootpath" : "~/Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"patcherrelativepath" : "../../../../Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"type" : "PNG",
				"implicit" : 1
			}
, 			{
				"name" : "python_env.txt",
				"bootpath" : "~/Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"patcherrelativepath" : "../../../../Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"type" : "TEXT",
				"implicit" : 1
			}
, 			{
				"name" : "acids.player.maxpat",
				"bootpath" : "~/Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"patcherrelativepath" : "../../../../Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "acids.button.maxpat",
				"bootpath" : "~/Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"patcherrelativepath" : "../../../../Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "micro.png",
				"bootpath" : "~/Dropbox/code/acids/team/axel/vschaos_package/max/plugin/graphics/icons_white_50",
				"patcherrelativepath" : "../../../../Dropbox/code/acids/team/axel/vschaos_package/max/plugin/graphics/icons_white_50",
				"type" : "PNG",
				"implicit" : 1
			}
, 			{
				"name" : "acids.audio.in.maxpat",
				"bootpath" : "~/Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"patcherrelativepath" : "../../../../Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "acids.fft.in.maxpat",
				"bootpath" : "~/Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"patcherrelativepath" : "../../../../Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "acids.sliders.maxpat",
				"bootpath" : "~/Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"patcherrelativepath" : "../../../../Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "acids.fft.out.maxpat",
				"bootpath" : "~/Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"patcherrelativepath" : "../../../../Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "acids.synth.fftsin.maxpat",
				"bootpath" : "~/Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"patcherrelativepath" : "../../../../Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "acids.synth.ifft.maxpat",
				"bootpath" : "~/Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"patcherrelativepath" : "../../../../Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "acids.model.drop.maxpat",
				"bootpath" : "~/Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"patcherrelativepath" : "../../../../Dropbox/code/acids/team/axel/vschaos_package/max/plugin",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "resume-button.png",
				"bootpath" : "~/Dropbox/code/acids/team/axel/vschaos_package/max/plugin/graphics/icons_white_50",
				"patcherrelativepath" : "../../../../Dropbox/code/acids/team/axel/vschaos_package/max/plugin/graphics/icons_white_50",
				"type" : "PNG",
				"implicit" : 1
			}
, 			{
				"name" : "mind-map-2.png",
				"bootpath" : "~/Dropbox/code/acids/team/axel/vschaos_package/max/plugin/graphics/icons_white_50",
				"patcherrelativepath" : "../../../../Dropbox/code/acids/team/axel/vschaos_package/max/plugin/graphics/icons_white_50",
				"type" : "PNG",
				"implicit" : 1
			}
, 			{
				"name" : "o.print.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "o.route.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "shell.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "o.pack.mxo",
				"type" : "iLaX"
			}
 ],
		"autosave" : 0,
		"styles" : [ 			{
				"name" : "AudioStatus_Menu",
				"default" : 				{
					"bgfillcolor" : 					{
						"type" : "color",
						"color" : [ 0.294118, 0.313726, 0.337255, 1 ],
						"color1" : [ 0.454902, 0.462745, 0.482353, 0.0 ],
						"color2" : [ 0.290196, 0.309804, 0.301961, 1.0 ],
						"angle" : 270.0,
						"proportion" : 0.39,
						"autogradient" : 0
					}

				}
,
				"parentstyle" : "",
				"multi" : 0
			}
, 			{
				"name" : "BlueTextButtons",
				"default" : 				{
					"bgcolor" : [ 0.666206, 0.782298, 0.817138, 1.0 ]
				}
,
				"parentstyle" : "RedTextButtons",
				"multi" : 0
			}
, 			{
				"name" : "BlueTextButtons-1",
				"default" : 				{
					"accentcolor" : [ 0.25, 0.25, 0.25, 1.0 ],
					"color" : [ 1.0, 1.0, 1.0, 1.0 ],
					"elementcolor" : [ 0.55, 0.55, 0.55, 1.0 ],
					"selectioncolor" : [ 0.1, 0.1, 0.1, 1.0 ],
					"fontname" : [ "Helvetica" ],
					"bgcolor" : [ 0.538741, 0.764449, 0.877768, 1.0 ]
				}
,
				"parentstyle" : "",
				"multi" : 0
			}
, 			{
				"name" : "GreenTextButtons",
				"parentstyle" : "RedTextButtons",
				"multi" : 0
			}
, 			{
				"name" : "Luca",
				"default" : 				{
					"bgfillcolor" : 					{
						"type" : "gradient",
						"color" : [ 0.290196, 0.309804, 0.301961, 1.0 ],
						"color1" : [ 0.862745, 0.870588, 0.878431, 1.0 ],
						"color2" : [ 0.65098, 0.666667, 0.662745, 1.0 ],
						"angle" : 270.0,
						"proportion" : 0.39,
						"autogradient" : 0
					}
,
					"accentcolor" : [ 0.32549, 0.345098, 0.372549, 1.0 ],
					"color" : [ 0.475135, 0.293895, 0.251069, 1.0 ],
					"elementcolor" : [ 0.786675, 0.801885, 0.845022, 1.0 ],
					"selectioncolor" : [ 0.720698, 0.16723, 0.080014, 1.0 ],
					"textcolor_inverse" : [ 0.239216, 0.254902, 0.278431, 1.0 ],
					"fontname" : [ "Open Sans Semibold" ],
					"bgcolor" : [ 0.904179, 0.895477, 0.842975, 0.56 ]
				}
,
				"parentstyle" : "",
				"multi" : 0
			}
, 			{
				"name" : "RedTextButtons",
				"default" : 				{
					"accentcolor" : [ 0.25, 0.25, 0.25, 1.0 ],
					"color" : [ 1.0, 1.0, 1.0, 1.0 ],
					"elementcolor" : [ 0.55, 0.55, 0.55, 1.0 ],
					"selectioncolor" : [ 0.1, 0.1, 0.1, 1.0 ],
					"fontname" : [ "Helvetica" ],
					"bgcolor" : [ 0.827321, 0.874747, 0.7195, 1.0 ]
				}
,
				"parentstyle" : "",
				"multi" : 0
			}
, 			{
				"name" : "RedTextButtons-1",
				"default" : 				{
					"accentcolor" : [ 0.25, 0.25, 0.25, 1.0 ],
					"color" : [ 1.0, 1.0, 1.0, 1.0 ],
					"elementcolor" : [ 0.55, 0.55, 0.55, 1.0 ],
					"selectioncolor" : [ 0.1, 0.1, 0.1, 1.0 ],
					"fontname" : [ "Helvetica" ],
					"bgcolor" : [ 0.843137, 0.733333, 0.729412, 1.0 ]
				}
,
				"parentstyle" : "",
				"multi" : 0
			}
, 			{
				"name" : "RedTextButtons-2",
				"default" : 				{
					"accentcolor" : [ 0.25, 0.25, 0.25, 1.0 ],
					"color" : [ 1.0, 1.0, 1.0, 1.0 ],
					"elementcolor" : [ 0.55, 0.55, 0.55, 1.0 ],
					"selectioncolor" : [ 0.1, 0.1, 0.1, 1.0 ],
					"fontname" : [ "Helvetica" ],
					"bgcolor" : [ 0.843137, 0.733333, 0.729412, 1.0 ]
				}
,
				"parentstyle" : "",
				"multi" : 0
			}
, 			{
				"name" : "RedTextButtons-3",
				"default" : 				{
					"accentcolor" : [ 0.25, 0.25, 0.25, 1.0 ],
					"color" : [ 1.0, 1.0, 1.0, 1.0 ],
					"elementcolor" : [ 0.55, 0.55, 0.55, 1.0 ],
					"selectioncolor" : [ 0.1, 0.1, 0.1, 1.0 ],
					"fontname" : [ "Helvetica" ],
					"bgcolor" : [ 0.843137, 0.733333, 0.729412, 1.0 ]
				}
,
				"parentstyle" : "",
				"multi" : 0
			}
, 			{
				"name" : "RedTextButtons-4",
				"default" : 				{
					"accentcolor" : [ 0.25, 0.25, 0.25, 1.0 ],
					"color" : [ 1.0, 1.0, 1.0, 1.0 ],
					"elementcolor" : [ 0.55, 0.55, 0.55, 1.0 ],
					"selectioncolor" : [ 0.1, 0.1, 0.1, 1.0 ],
					"fontname" : [ "Helvetica" ],
					"bgcolor" : [ 0.843137, 0.733333, 0.729412, 1.0 ]
				}
,
				"parentstyle" : "",
				"multi" : 0
			}
, 			{
				"name" : "RedTextButtons-5",
				"default" : 				{
					"accentcolor" : [ 0.25, 0.25, 0.25, 1.0 ],
					"color" : [ 1.0, 1.0, 1.0, 1.0 ],
					"elementcolor" : [ 0.55, 0.55, 0.55, 1.0 ],
					"selectioncolor" : [ 0.1, 0.1, 0.1, 1.0 ],
					"fontname" : [ "Helvetica" ],
					"bgcolor" : [ 0.843137, 0.733333, 0.729412, 1.0 ]
				}
,
				"parentstyle" : "",
				"multi" : 0
			}
, 			{
				"name" : "RedTextButtons-6",
				"default" : 				{
					"accentcolor" : [ 0.25, 0.25, 0.25, 1.0 ],
					"color" : [ 1.0, 1.0, 1.0, 1.0 ],
					"elementcolor" : [ 0.55, 0.55, 0.55, 1.0 ],
					"selectioncolor" : [ 0.1, 0.1, 0.1, 1.0 ],
					"fontname" : [ "Helvetica" ],
					"bgcolor" : [ 0.843137, 0.733333, 0.729412, 1.0 ]
				}
,
				"parentstyle" : "",
				"multi" : 0
			}
, 			{
				"name" : "VioletTextButton",
				"default" : 				{
					"accentcolor" : [ 0.25, 0.25, 0.25, 1.0 ],
					"color" : [ 1.0, 1.0, 1.0, 1.0 ],
					"elementcolor" : [ 0.55, 0.55, 0.55, 1.0 ],
					"selectioncolor" : [ 0.1, 0.1, 0.1, 1.0 ],
					"fontname" : [ "Helvetica" ],
					"bgcolor" : [ 0.715377, 0.696413, 0.824482, 1.0 ]
				}
,
				"parentstyle" : "",
				"multi" : 0
			}
, 			{
				"name" : "dark-night-patch",
				"default" : 				{
					"bgfillcolor" : 					{
						"type" : "gradient",
						"color1" : [ 0.376471, 0.384314, 0.4, 1.0 ],
						"color2" : [ 0.290196, 0.309804, 0.301961, 1.0 ],
						"color" : [ 0.290196, 0.309804, 0.301961, 1.0 ],
						"angle" : 270.0,
						"proportion" : 0.39
					}
,
					"accentcolor" : [ 0.952941, 0.564706, 0.098039, 1.0 ],
					"textcolor" : [ 0.862745, 0.870588, 0.878431, 1.0 ],
					"patchlinecolor" : [ 0.439216, 0.74902, 0.254902, 0.898039 ]
				}
,
				"parentstyle" : "",
				"multi" : 0
			}
, 			{
				"name" : "jpatcher001",
				"parentstyle" : "",
				"multi" : 0
			}
, 			{
				"name" : "ksliderWhite",
				"default" : 				{
					"color" : [ 1.0, 1.0, 1.0, 1.0 ]
				}
,
				"parentstyle" : "",
				"multi" : 0
			}
, 			{
				"name" : "maty@multislider001",
				"parentstyle" : "multislider001",
				"multi" : 0
			}
, 			{
				"name" : "maty_jpatcher01",
				"default" : 				{
					"fontname" : [ "Helvetica Neue Thin" ]
				}
,
				"parentstyle" : "jpatcher001",
				"multi" : 0
			}
, 			{
				"name" : "maty_multislider01",
				"parentstyle" : "multislider001",
				"multi" : 0
			}
, 			{
				"name" : "multislider001",
				"default" : 				{
					"color" : [ 0.0, 0.0, 0.0, 1.0 ],
					"bgcolor" : [ 0.945827, 0.711942, 0.174445, 0.0 ]
				}
,
				"parentstyle" : "",
				"multi" : 0
			}
, 			{
				"name" : "newobjBlue-1",
				"default" : 				{
					"accentcolor" : [ 0.317647, 0.654902, 0.976471, 1.0 ]
				}
,
				"parentstyle" : "",
				"multi" : 0
			}
, 			{
				"name" : "newobjGreen-1",
				"default" : 				{
					"accentcolor" : [ 0.0, 0.533333, 0.168627, 1.0 ]
				}
,
				"parentstyle" : "",
				"multi" : 0
			}
, 			{
				"name" : "numberGold-1",
				"default" : 				{
					"accentcolor" : [ 0.764706, 0.592157, 0.101961, 1.0 ]
				}
,
				"parentstyle" : "",
				"multi" : 0
			}
 ],
		"bgcolor" : [ 0.349019607843137, 0.349019607843137, 0.349019607843137, 1.0 ]
	}

}
